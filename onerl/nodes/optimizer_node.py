import pickle
import multiprocessing as mp
import ctypes
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from onerl.utils.import_module import get_class_from_str
from onerl.utils.batch.cuda import BatchCuda
from onerl.nodes.node import Node


class OptimizerNode(Node):
    @staticmethod
    def create_algo(ns_config: dict):
        algo_config = ns_config["algorithm"]
        # create network
        network = {k: get_class_from_str(v.get("import", ""), v["name"])(**v.get("params", {}))
                   for k, v in algo_config.get("network", {}).items()}

        algo_class = get_class_from_str(algo_config.get("import", ""), algo_config["name"])
        return algo_class(network=network, env_params=ns_config["env"], **algo_config.get("params", {}))

    @staticmethod
    def node_preprocess_ns_config(node_class: str, num: int, ns_config: dict):
        Node.node_preprocess_ns_config(node_class, num, ns_config)

        # model size
        algo = OptimizerNode.create_algo(ns_config)
        ns_config["algorithm"]["pickled_size"] = len(pickle.dumps(algo.state_dict()))

    @staticmethod
    def node_create_shared_objects(node_class: str, num: int, ns_config: dict):
        objects = Node.node_create_shared_objects(node_class, num, ns_config)
        # rank 0 only, policy update
        objects[0].update({
            "update_lock": mp.Lock(),
            "update_version": mp.Value(ctypes.c_int64, -1, lock=False),
            "update_state": mp.Array(ctypes.c_uint8, ns_config["algorithm"]["pickled_size"], lock=False)
        })
        return objects

    def run(self):
        # distributed data parallel (DDP)
        # setup DDP
        # FIXME: Single machine multi-GPU setting
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = self.config.get("port", "12355")
        dist.init_process_group("nccl", rank=self.node_rank,
                                world_size=self.node_count(self.node_class, self.ns_config))
        # allocate device
        devices = self.config["devices"]
        device = torch.device(devices[self.node_rank % len(devices)])
        # model
        model = self.create_algo(self.ns_config).to(device)
        model = DistributedDataParallel(model, device_ids=[device])

        # updater
        last_update_time = time.time()
        current_model_version = 0

        # optimizer
        node_sampler = self.find("SamplerNode", self.node_rank)
        batch = BatchCuda(self.ns_config[node_sampler]["batch"], device)
        # sample first batch
        self.send(node_sampler, "")

        while True:
            # wait & copy batch
            self.setstate("wait")
            batch.wait_ready()
            self.setstate("copy")
            batch.copy_from()
            # notify to sample
            self.send(node_sampler, "")

            # optimize
            self.setstate("step")
            model.learn(batch)

            # update (if needed)
            if self.node_rank == 0:
                current_model_version += 1
                current_time = time.time()
                if (current_time - last_update_time) >= self.config["update_interval"]:
                    self.setstate("update")
                    # serialize
                    state_dict_str = pickle.dumps(model.state_dict())
                    # update shared
                    self.objects["update_lock"].acquire()
                    self.objects["update_version"].value = current_model_version
                    self.objects["update_state"][:] = state_dict_str
                    self.objects["update_lock"].release()
                    # release serialized model
                    del state_dict_str
