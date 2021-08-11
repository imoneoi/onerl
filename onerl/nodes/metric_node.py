import multiprocessing as mp
import ctypes

from onerl.nodes.node import Node


class MetricNode(Node):
    @staticmethod
    def node_create_shared_objects(node_class: str, num: int, global_config: dict):
        objects = Node.node_create_shared_objects(node_class, num, global_config)

        assert num == 1, "MetricNode: There must be only one metric node."
        objects[0].update({
            "lock": mp.Lock(),
            "tick": mp.Value(ctypes.c_int64, 0, lock=False)
        })
        return objects

    def run(self):
        while True:
            self.recv()
