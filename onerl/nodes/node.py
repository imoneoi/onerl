import time
import os

from faster_fifo import Queue

import setproctitle


class Node:
    # Node basics
    def __init__(self, node_class: str, node_ns: str, node_rank: int, node_config: dict,
                 ns_config: dict, peer_objects: dict):
        super().__init__()
        self.node_ns = node_ns
        self.node_class = node_class
        self.node_rank = node_rank
        self.node_name = self.get_node_name(node_ns, node_class, node_rank)
        self.config = node_config

        self.ns_config = ns_config
        self.peer_objects = peer_objects

        self.objects = self.peer_objects["self"]
        self.queue = self.objects["queue"]

        # State & profiling
        self.is_profile = self.ns_config.get("profile", False)
        if self.is_profile:
            profile_log_filename = os.path.join(self.ns_config["profile_log_path"], self.node_name)
            self.profile_stream = open(profile_log_filename, "wb",
                                       buffering=self.ns_config.get("profile_log_buffer", 1048576))

        # Proc title for visualization
        setproctitle.setproctitle("-OneRL- {}".format(self.node_name))

    # torch utilities
    def setup_torch_opt(self):
        import torch

        # CUDNN Benchmark
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        # Limit torch to 1 thread
        torch.set_num_threads(1)

    # Node methods
    @staticmethod
    def node_preprocess_ns_config(node_class: str, num: int, ns_config: dict):
        ns_config.setdefault("num", {})
        ns_config["num"][node_class] = num

    @staticmethod
    def node_create_shared_objects(node_class: str, num: int, ns_config: dict):
        # create queues
        return [{"queue": Queue(max_size_bytes=1048576)} for _ in range(num)]

    @staticmethod
    def node_import_peer_objects(node_class: str, num: int, ns_config: dict, ns_objects: dict, all_ns_objects: dict):
        return [{
            "self": ns_objects[node_class][rank],

            "metric": ns_objects.get("MetricNode")
        } for rank in range(num)]

    # Node name convention
    @staticmethod
    def get_node_name(node_ns: str, node_class: str, node_rank: int):
        return "{}@{}.{}".format(node_ns, node_class, node_rank)  # Naming convention

    # Node utils
    @staticmethod
    def node_count(node_class: str, ns_config: dict):
        return ns_config["num"].get(node_class, 0)

    # State / Profiling
    def setstate(self, state: str):
        if self.is_profile:
            self.profile_stream.write(time.time_ns().to_bytes(8, "big") + state.encode() + b"\0")

    # Comm
    def has_peer(self, peer: str):
        return self.peer_objects.get(peer) is not None

    def send(self, peer: str, rank: int, msg: any):
        self.peer_objects[peer][rank]["queue"].put(msg)

    def recv(self):
        return self.queue.get(timeout=1000000000)

    def recv_all(self):
        return self.queue.get_many(timeout=1000000000)

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
        if self.has_peer("metric"):
            self.send("metric", 0, metric)
