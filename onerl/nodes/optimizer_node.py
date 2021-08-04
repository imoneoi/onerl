import pickle
import multiprocessing as mp
import ctypes
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from onerl.utils.import_module import get_class_from_str
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
    def node_preprocess_global_config(global_config: dict):
        super().node_preprocess_global_config(global_config)

        # model size
        algo = OptimizerNode.create_algo(global_config)
        global_config["algorithm"]["pickled_size"] = len(pickle.dumps(algo.state_dict()))

    @staticmethod
    def node_create_shared_objects(num: int, global_config: dict):
        objects = super().node_create_shared_objects(num, global_config)
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
        dist.init_process_group("nccl", rank=self.node_rank, world_size=self.count_nodes(self.node_class))
        # allocate device
        devices = self.config["devices"]
        device = devices[self.node_rank % len(devices)]
        # model
        model = self.create_algo(self.global_config).to(device)
        model = DDP(model, device_ids=[device])

        # optimizer
        while True:
