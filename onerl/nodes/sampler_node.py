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
            }, init_ready=False)

        return objects

    def run(self):
        # replay buffer shared objs
        replay_buffer_objs = self.global_objects[self.get_node_name("ReplayBufferNode", 0)]

        shared_buffer = replay_buffer_objs["buffer"].get()
        shared_size = replay_buffer_objs["size"].get()
        shared_idx = replay_buffer_objs["idx"].get()
        shared_lock = replay_buffer_objs["lock"]
        num_buffers = shared_buffer.shape[0]

        # batch
        shared_batch = self.objects["batch"].get()
        batch_keys = list(shared_batch.__dict__.keys())

        batch_size = self.global_config["algorithm"]["params"]["batch_size"]
        frame_stack = self.global_config["env"]["frame_stack"] + 1

        # local idx & size (for lock-free)
        size = np.zeros_like(shared_size)
        idx = np.zeros_like(shared_idx)
        protect_range = self.config.get("protect_range", 10) + frame_stack

        # warm start
        self.setstate("warm_start")
        while True:
            # atomic copy idx
            shared_lock.acquire()
            size[:] = shared_size
            shared_lock.release()
            if np.min(size) > protect_range:
                break

        # event loop
        while True:
            # wait request
            self.setstate("wait_request")
            self.recv()

            self.setstate("calc_idx")
            # copy batch (lock-free)
            # atomic copy idx
            shared_lock.acquire()
            size[:] = shared_size
            idx[:] = shared_idx
            shared_lock.release()

            # sample (ignore idx ... idx + protect range)
            sample_start = (idx + protect_range) % size
            sample_len = size - protect_range
            sample_idx = (sample_start +
                          np.random.randint(0, sample_len, (batch_size // num_buffers, num_buffers))) % size
            # batch query
            # sample_idx: (N_idx, N_buf) -transpose-> (N_buf, N_idx)
            sample_idx = sample_idx.transpose().flatten()
            buf_idx = np.repeat(np.arange(num_buffers), batch_size // num_buffers)

            # copy
            self.setstate("copy")
            for k in batch_keys:
                shared_batch.__dict__[k] = shared_buffer.__dict__[k][buf_idx, sample_idx]
