import multiprocessing as mp
from socket import timeout
import time
import os

import torch

from faster_fifo import Queue
import faster_fifo_reduction

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
                                       buffering=self.ns_config.get("profile_log_buffer", 100 * 1024))

        # Metric
        self.metric_node = self.find("MetricNode", 0)

        # Proc title for visualization
        setproctitle.setproctitle("-OneRL- {}".format(self.node_name))
        # CUDNN Benchmark
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
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
        return [{"queue": Queue(max_size_bytes=1048576)} for _ in range(num)]

    # Node utils
    @staticmethod
    def node_count(node_class: str, ns_config: dict):
        return ns_config["num"].get(node_class, 0)

    # State
    def setstate(self, state: str):
        if self.is_profile:
            self.profile_stream.write(time.time_ns().to_bytes(8, "big") + state.encode() + b"\0")

    # Comm
    def find(self, node_class: str, node_rank: int = 0, target_ns: str = None):
        # find in local namespace
        name = self.get_node_name(target_ns if target_ns is not None else self.node_ns, node_class, node_rank)
        if name in self.global_objects:
            return name
        # then global namespace
        # FIXME: No global namespace in future
        name = self.get_node_name("$global", node_class, node_rank)
        if name in self.global_objects:
            return name
        return None

    def find_all(self, node_class: str):
        node_names = []
        for rank in range(self.node_count(node_class, self.ns_config)):
            node_names.append(self.get_node_name(self.node_ns, node_class, rank))

        return node_names

    # Queue
    def send(self, target_name: str, msg: any):
        self.global_objects[target_name]["queue"].put(msg)

    def recv(self):
        return self.queue.get(timeout=180)

    def recv_all(self):
        return self.queue.get_many(timeout=180)

    def available(self):
        return not self.queue.empty()

    # Run
    def run(self):
        assert False, "Node {} run not implemented".format(self.node_name)

    # Logging
    def log(self, *args, **kwargs):
        print(self.node_name, *args, **kwargs)

    # Metric
    def log_metric(self, metric):
        if self.metric_node is not None:
            self.send(self.metric_node, metric)
