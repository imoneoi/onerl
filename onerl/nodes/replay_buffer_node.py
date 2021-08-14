import multiprocessing as mp
import ctypes

import numpy as np

from onerl.nodes.node import Node
from onerl.utils.shared_array import SharedArray
from onerl.utils.batch.shared import BatchShared


class ReplayBufferNode(Node):
    @staticmethod
    def node_create_shared_objects(node_class: str, num: int, ns_config: dict):
        objects = Node.node_create_shared_objects(node_class, num, ns_config)

        assert num == 1, "Only one ReplayBufferNode is supported."
        # size
        total_size = ns_config["algorithm"]["params"]["replay_buffer_size"]
        num_buffers = Node.node_count("EnvNode", ns_config)
        single_size = total_size // num_buffers
        # create buffers
        objects[0].update({
            "buffer": BatchShared({
                k: ((num_buffers, single_size, *batch_shape), batch_dtype)
                for k, (batch_shape, batch_dtype) in ns_config["env"]["batch"].items()
            }),
            "size": SharedArray(num_buffers, dtype=np.int64),
            "idx": SharedArray(num_buffers, dtype=np.int64),
            "lock": mp.Lock()
        })
        return objects

    def run(self):
        # shared array
        shared_buffer = self.objects["buffer"].get()
        shared_size = self.objects["size"].get()
        shared_idx = self.objects["idx"].get()
        shared_lock = self.objects["lock"]
        # shared (remote)
        env_prefix = "{}@EnvNode.".format(self.node_ns)
        shared_env_log = {int(k[len(env_prefix):]): v["log"].get()
                          for k, v in self.global_objects.items() if k.startswith(env_prefix)}

        # event loop
        buffer_keys = list(shared_buffer.__dict__.keys())
        single_buffer_size = shared_buffer.__dict__[buffer_keys[0]].shape[1]
        while True:
            self.setstate("wait")
            env_name = self.recv()
            env_id = int(env_name[len(env_prefix):])

            # idx & size
            self.setstate("copy")
            idx = shared_idx[env_id]
            size = shared_size[env_id]
            # copy buffer
            for k in buffer_keys:
                shared_buffer.__dict__[k][env_id, idx] = shared_env_log[env_id].__dict__[k]
            self.global_objects[env_name]["log"].set_ready()

            # move index
            self.setstate("move_index")
            new_idx = (idx + 1) % single_buffer_size
            new_size = min(single_buffer_size, size + 1)

            shared_lock.acquire()
            shared_idx[env_id] = new_idx
            shared_size[env_id] = new_size
            shared_lock.release()
