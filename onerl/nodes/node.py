import multiprocessing as mp


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

        self.state = None

    # Node methods
    @staticmethod
    def get_node_name(node_class: str, node_rank: int):
        return "{}.{}".format(node_class, node_rank)  # Naming convention

    @staticmethod
    def node_preprocess_global_config(global_config: dict):
        pass

    @staticmethod
    def node_create_shared_objects(num: int, global_config: dict):
        # create queues
        return [{"queue": mp.SimpleQueue()} for _ in range(num)]

    # State
    def setstate(self, state: str):
        self.state = state

    # Queue
    def send(self, target_name: str, msg: any):
        self.global_objects[target_name]["queue"].put(msg)

    def recv(self):
        return self.queue.get()

    def available(self):
        return not self.queue.empty()

    # Run
    def dummy_init(self):
        if self.config.get("dummy", False):
            # dummy loop for debugging
            self.setstate("dummy")
            while True:
                self.recv()

    def run(self):
        assert False, "Node {} run not implemented".format(self.node_name)
