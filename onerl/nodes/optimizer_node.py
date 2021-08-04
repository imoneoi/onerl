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
    def create_algo(global_config):
        algo_config = global_config["algorithm"]
        # create network
        network = {k: get_class_from_str(v.get("import", ""), v["name"])(**v.get("params", {}))
                   for k, v in algo_config["network"].items()}

        algo_class = get_class_from_str(algo_config.get("import", ""), algo_config["name"])
        return algo_class(network=network, **algo_config.get("params", {}))

    @staticmethod
    def node_preprocess_global_config(node_class: str, num: int, global_config: dict):
        super().node_preprocess_global_config(node_class, num, global_config)

        # model size
        algo = OptimizerNode.create_algo(global_config)
        global_config["algorithm"]["pickled_size"] = len(pickle.dumps(algo.state_dict()))

    @staticmethod
    def node_create_shared_objects(node_class: str, num: int, global_config: dict):
        objects = super().node_create_shared_objects(node_class, num, global_config)
        # rank 0 only, policy update
        objects[0].update({
            "update_lock": mp.Lock(),
            "update_version": mp.Value(ctypes.c_int64, 0, lock=False),
            "update_state": mp.Array(ctypes.c_uint8, global_config["algorithm"]["pickled_size"], lock=False)
        })
        return objects

    def run(self):
        self.dummy_init()

        # distributed data parallel (DDP)
        # setup DDP
        # FIXME: Single machine multi-GPU setting
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = self.config.get("port", "12355")
        dist.init_process_group("nccl", rank=self.node_rank,
                                world_size=self.node_count_by_class(self.node_class, self.global_config))
        # allocate device
        devices = self.config["devices"]
        device = torch.device(devices[self.node_rank % len(devices)])
        # model
        model = self.create_algo(self.global_config).to(device)
        model = DistributedDataParallel(model, device_ids=[device])

        # updater
        last_update_time = time.time()
        current_model_version = 0

        # optimizer
        sampler_name = self.get_node_name("SamplerNode", 0)
        batch = BatchCuda(self.global_config[sampler_name]["batch_{}".format(self.node_rank)], device)
        # sample first batch
        self.send(sampler_name, self.node_rank)

        while True:
            # wait & copy batch
            self.setstate("wait_batch")
            batch.wait_ready()
            self.setstate("copy_batch")
            batch.copy_from()
            # notify to sample
            self.send(sampler_name, self.node_rank)

            # optimize
            self.setstate("optimize")
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
