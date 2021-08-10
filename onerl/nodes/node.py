import multiprocessing as mp
import time
import os

import torch

import setproctitle


class Node:
    # Node basics
    def __init__(self, node_class: str, node_ns: str, node_rank: int, node_config: dict,
                 ns_config: dict, global_objects: dict):
        super().__init__()
        self.node_ns = node_ns
        self.node_class = node_class
        self.node_rank = node_rank
        self.node_name = self.get_node_name(node_ns, node_class, node_rank)
        self.config = node_config

        self.ns_config = ns_config
        self.global_objects = global_objects

        self.objects = self.global_objects[self.node_name]
        self.queue = self.objects["queue"]

        # State & profiling
        self.is_profile = self.ns_config.get("profile", False)
        if self.is_profile:
            profile_log_filename = os.path.join(self.ns_config["profile_log_path"], self.node_name)
            self.profile_stream = open(profile_log_filename, "wb",
                                       buffering=self.ns_config.get("profile_log_buffer", 1048576))

        # Proc title for visualization
        setproctitle.setproctitle("-OneRL- {}".format(self.node_name))
        # Limit torch to 1 thread
        torch.set_num_threads(1)

    # Node methods
    @staticmethod
    def get_node_name(node_ns: str, node_class: str, node_rank: int):
        return "{}@{}.{}".format(node_ns, node_class, node_rank)  # Naming convention

    @staticmethod
    def node_preprocess_ns_config(node_class: str, num: int, ns_config: dict):
        ns_config.setdefault("num", {})
        ns_config["num"][node_class] = num

    @staticmethod
    def node_create_shared_objects(node_class: str, num: int, ns_config: dict):
        # create queues
        return [{"queue": mp.SimpleQueue()} for _ in range(num)]

    # Node utils
    @staticmethod
    def node_count(node_class: str, ns_config: dict):
        return ns_config["num"][node_class]

    # State
    def setstate(self, state: str):
        if self.is_profile:
            self.profile_stream.write(time.time_ns().to_bytes(8, "big") + state.encode() + b"\0")

    # Comm
    def find(self, node_class: str, node_rank: int = 0):
        # find in local namespace
        name = self.get_node_name(self.node_ns, node_class, node_rank)
        if name in self.global_objects:
            return name
        # then global namespace
        name = self.get_node_name("", node_class, node_rank)
        if name in self.global_objects:
            return name

        return None

    def send(self, target_name: str, msg: any):
        self.global_objects[target_name]["queue"].put(msg)

    def recv(self):
        return self.queue.get()

    def available(self):
        return not self.queue.empty()

    # Run
    def run(self):
        assert False, "Node {} run not implemented".format(self.node_name)
