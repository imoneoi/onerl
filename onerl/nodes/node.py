import multiprocessing as mp
import time
import os

import torch

import setproctitle


class Node:
    # Node basics
    def __init__(self, node_class: str, node_rank: int, node_config: dict,
                 global_config: dict, global_objects: dict):
        super().__init__()
        self.node_class = node_class
        self.node_rank = node_rank
        self.node_name = self.get_node_name(node_class, node_rank)
        self.config = node_config

        self.global_config = global_config
        self.global_objects = global_objects

        self.objects = self.global_objects[self.node_name]
        self.queue = self.objects["queue"]

        # State & profiling
        self.is_profile = global_config.get("profile", False)
        if self.is_profile:
            profile_log_filename = os.path.join(global_config["profile_log_path"], self.node_name)
            self.profile_stream = open(profile_log_filename, "wb",
                                       buffering=global_config.get("profile_log_buffer", 1048576))

        # Metric
        self.metric_node = self.get_node_name("MetricNode", 0)
        if self.metric_node not in self.global_objects:
            self.metric_node = None

        # Proc title for visualization
        setproctitle.setproctitle("-OneRL- {}".format(self.node_name))
        # Limit torch to 1 thread
        torch.set_num_threads(1)

    # Node methods
    @staticmethod
    def get_node_name(node_class: str, node_rank: int):
        return "{}.{}".format(node_class, node_rank)  # Naming convention

    @staticmethod
    def node_preprocess_global_config(node_class: str, num: int, global_config: dict):
        global_config.setdefault("num", {})
        global_config["num"][node_class] = num

    @staticmethod
    def node_create_shared_objects(node_class: str, num: int, global_config: dict):
        # create queues
        return [{"queue": mp.SimpleQueue()} for _ in range(num)]

    # Node utils
    @staticmethod
    def node_count_by_class(node_class: str, global_config: dict):
        return global_config["num"][node_class]

    # State
    def setstate(self, state: str):
        if self.is_profile:
            self.profile_stream.write(time.time_ns().to_bytes(8, "big") + state.encode() + b"\0")

    # Queue
    def send(self, target_name: str, msg: any):
        self.global_objects[target_name]["queue"].put(msg)

    def recv(self):
        return self.queue.get()

    def available(self):
        return not self.queue.empty()

    # Run
    def init(self):
        if self.config.get("dummy", False):
            # dummy loop for debugging
            self.setstate("dummy")
            self.run_dummy()

    def run(self):
        assert False, "Node {} run not implemented".format(self.node_name)

    def run_dummy(self):
        while True:
            self.recv()

    # Metric
    def log_metric(self, metric):
        if self.metric_node is not None:
            self.send(self.metric_node, metric)
