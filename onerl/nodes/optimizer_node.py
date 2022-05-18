import pickle
import multiprocessing as mp
import ctypes
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from onerl.utils.import_module import get_class_from_str
from onerl.utils.shared_state_dict import SharedStateDict
from onerl.utils.batch.cuda import BatchCuda
from onerl.nodes.node import Node


class OptimizerNode(Node):
    @staticmethod
    def create_algo(ns_config: dict, net_device: torch.device = None, use_ddp: bool = False):
        algo_config = ns_config["algorithm"]
        # create network
        network = {k: get_class_from_str(v.get("import", ""), v["name"])(**v.get("params", {}))
                   for k, v in algo_config.get("network", {}).items()}
        if net_device is not None:
            device_ids = [net_device] if net_device.type != "cpu" else None
            network = {k: DistributedDataParallel(v.to(net_device), device_ids=device_ids)
                       if use_ddp and len(list(v.parameters())) else v.to(net_device)
                       for k, v in network.items()}

        algo_class = get_class_from_str(algo_config.get("import", ""), algo_config["name"])
        return algo_class(network=network, env_params=ns_config["env"], **algo_config.get("params", {}))

    @staticmethod
    def node_create_shared_objects(node_class: str, num: int, ns_config: dict):
        objects = Node.node_create_shared_objects(node_class, num, ns_config)
        # policy state dict example
        example_policy_state_dict = OptimizerNode.create_algo(ns_config).policy_state_dict()
        # rank 0 only, policy update
        objects[0].update({
            "update_lock": mp.Lock(),
            "update_version": mp.RawValue(ctypes.c_int64, -1),
            "update_state_dict": SharedStateDict(example_policy_state_dict)
        })
        return objects

    @staticmethod
    def node_import_peer_objects(node_class: str, num: int, ns_config: dict, ns_objects: dict, all_ns_objects: dict):
        objects = Node.node_import_peer_objects(node_class, num, ns_config, ns_objects, all_ns_objects)
        for obj in objects:
            obj["sampler"] = ns_objects.get("SamplerNode")

        return objects

    def run(self):
        self.setup_torch_opt()  # setup torch optimizations

        # allocate device
        devices = self.config["devices"]
        device = torch.device(devices[self.node_rank % len(devices)])
        is_cpu = device.type == "cpu"

        # distributed data parallel (DDP)
        use_ddp = False
        if self.node_count("OptimizerNode", self.ns_config) > 1:
            use_ddp = True
            # setup DDP
            # FIXME: Single machine multi-GPU setting
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = self.config.get("port", "12355")

            # GLOO for CPU training, NCCL for GPU training
            # https://pytorch.org/docs/stable/distributed.html
            if is_cpu:
                dist_backend = "gloo"
            else:
                dist_backend = "nccl"
            dist.init_process_group(dist_backend, rank=self.node_rank,
                                    world_size=self.node_count(self.node_class, self.ns_config))

        # model
        algorithm = self.create_algo(self.ns_config, device, use_ddp)
        algorithm.train()

        # ticks
        shared_tick = self.peer_objects["metric"][0]["tick"]

        # updater
        last_update_time = time.time()
        current_model_version = 0

        local_update_state_dict = None
        shared_update_state_dict = None
        if self.node_rank == 0:
            local_update_state_dict = algorithm.policy_state_dict()
            shared_update_state_dict = self.objects["update_state_dict"]
            shared_update_state_dict.initialize("publisher", device)

        # save model
        last_save_model_time = time.time()
        save_model_path = self.config.get("save_path", "models")
        os.makedirs(save_model_path, exist_ok=True)

        # optimizer
        batch = BatchCuda(self.peer_objects["sampler"][self.node_rank]["batch"], device)
        # sample first batch
        self.send("sampler", self.node_rank, "")

        while True:
            # wait & copy batch
            self.setstate("wait")
            batch.wait_ready()
            self.setstate("copy")
            batch.copy_from()
            # notify to sample
            self.send("sampler", self.node_rank, "")

            # optimize
            self.setstate("step")
            metric = algorithm.learn(batch, shared_tick.value)
            if metric is not None:
                # update to data logging
                if self.node_rank == 0:
                    metric["update"] = 1

                self.log_metric(metric)

            if self.node_rank == 0:
                # update (if needed)
                current_model_version += 1
                current_time = time.time()
                if (current_time - last_update_time) >= self.config["update_interval"]:
                    last_update_time = current_time

                    # update shared policy (lock free)
                    self.setstate("update_policy")
                    shared_update_state_dict.publish(local_update_state_dict)

                    self.objects["update_lock"].acquire()
                    self.objects["update_version"].value = current_model_version
                    self.objects["update_lock"].release()

                # save model
                if (current_time - last_save_model_time) >= self.config.get("save_interval", 3600):
                    last_save_model_time = current_time

                    save_filename = os.path.join(save_model_path, str(current_model_version))
                    torch.save(algorithm.state_dict(), save_filename)
                    # log model
                    self.log_metric({"save_model": True, "save_filename": save_filename})
