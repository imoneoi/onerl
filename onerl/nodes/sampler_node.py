import numpy as np

from onerl.nodes.node import Node
from onerl.utils.batch.shared import BatchShared


class SamplerNode(Node):
    @staticmethod
    def node_create_shared_objects(node_class: str, num: int, global_config: dict):
        objects = Node.node_create_shared_objects(node_class, num, global_config)

        assert Node.node_count_by_class("OptimizerNode", global_config) == num, \
               "Number of sampler nodes must equal to number of optimizer nodes."

        batch_size = global_config["algorithm"]["params"]["batch_size"]
        frame_stack = global_config["env"]["frame_stack"] + 1
        for obj in objects:
            obj["batch"] = BatchShared({
                "obs": ((batch_size, frame_stack, *global_config["env"]["obs_shape"]),
                        global_config["env"]["obs_dtype"]),
                "rew": ((batch_size, frame_stack), np.float32),
                "done": ((batch_size, frame_stack), np.bool_)
            })

        return objects

    def run(self):
        pass
