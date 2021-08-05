import multiprocessing as mp
import ctypes

import numpy as np

from onerl.nodes.node import Node
from onerl.utils.shared_array import SharedArray
from onerl.utils.batch.shared import BatchShared


class ReplayBufferNode(Node):
    @staticmethod
    def node_create_shared_objects(node_class: str, num: int, global_config: dict):
        objects = Node.node_create_shared_objects(node_class, num, global_config)

        assert num == 1, "Only one ReplayBufferNode is supported."
        # size
        total_size = global_config["algorithm"]["params"]["replay_buffer_size"]
        num_buffers = Node.node_count_by_class("EnvNode", global_config)
        single_size = total_size // num_buffers
        # create buffers
        objects[0].update({
            "buffer": [
                BatchShared({
                    "obs": ((single_size, *global_config["env"]["obs_shape"]), global_config["env"]["obs_dtype"]),
                    "rew": ((single_size, ), np.float32),
                    "done": ((single_size, ), np.bool_)
                }) for _ in range(num_buffers)
            ],
            "size": [mp.Value(ctypes.c_int64, lock=False) for _ in range(num_buffers)],
            "idx": [mp.Value(ctypes.c_int64, lock=False) for _ in range(num_buffers)],
            "lock": [mp.Lock() for _ in range(num_buffers)]
        })
        return objects

    def run(self):
        # shared array
        shared_buffer = [item.get() for item in self.objects["buffer"]]
        shared_size = self.objects["size"]
        shared_idx = self.objects["idx"]
        shared_lock = self.objects["lock"]
        # shared (remote)
        shared_env_log = {int(k[len("EnvNode."):]): v["log"].get()
                          for k, v in self.global_objects.items() if k.startswith("EnvNode.")}

        # event loop
        buffer_keys = list(shared_buffer[0].__dict__.keys())
        single_buffer_size = shared_buffer[0].__dict__[buffer_keys[0]].shape[0]
        while True:
            env_name = self.recv()
            env_id = int(env_name[len("EnvNode."):])
            # idx & size
            idx = shared_idx[env_id]
            size = shared_size[env_id]

            # copy buffer
            for k in buffer_keys:
                shared_buffer[env_id].__dict__[k][idx.value] = shared_env_log[env_id].__dict__[k]
            self.global_objects[env_name]["log"].set_ready()

            # move index
            new_idx = (idx.value + 1) % single_buffer_size
            new_size = min(single_buffer_size, size.value + 1)

            shared_lock[env_id].acquire()
            idx.value = new_idx
            size.value = new_size
            shared_lock[env_id].release()

    def run_dummy(self):
        while True:
            env_name = self.recv()
            # dummy log (no)
            self.global_objects[env_name]["log"].set_ready()
