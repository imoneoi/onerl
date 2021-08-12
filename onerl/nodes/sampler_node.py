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
                k: ((batch_size, frame_stack, *batch_shape), batch_dtype)
                for k, (batch_shape, batch_dtype) in global_config["env"]["batch"].items()
            }, init_ready=False)

        return objects

    def run(self):
        # replay buffer shared objs
        replay_buffer_objs = self.global_objects[self.get_node_name("ReplayBufferNode", 0)]

        # local idx & size (for lock-free)
        shared_buffer = replay_buffer_objs["buffer"].get()

        shared_lock = replay_buffer_objs["lock"]
        shared_size = replay_buffer_objs["size"].get()
        shared_idx = replay_buffer_objs["idx"].get()

        local_size = np.zeros_like(shared_size)
        local_idx = np.zeros_like(shared_idx)

        # shapes
        num_buffers = list(shared_buffer.__dict__.values())[0].shape[0]

        # batch & batch params
        shared_batch = self.objects["batch"].get()
        batch_keys = list(shared_batch.__dict__.keys())

        batch_size = self.global_config["algorithm"]["params"]["batch_size"]
        frame_stack = self.global_config["env"]["frame_stack"] + 1

        # lock-free params
        protect_range = self.config.get("protect_range", 10) + frame_stack

        # warm start
        self.setstate("start")
        while True:
            # atomic copy idx
            shared_lock.acquire()
            local_size[:] = shared_size
            shared_lock.release()
            if np.min(local_size) > protect_range:
                break

        # event loop
        while True:
            # wait request
            self.setstate("wait")
            self.recv()

            self.setstate("calc_index")
            # copy batch (lock-free)
            # atomic copy idx
            shared_lock.acquire()
            local_size[:] = shared_size
            local_idx[:] = shared_idx
            shared_lock.release()

            # sample (ignore idx ... idx + protect range)
            # shape (N, N_Buffer)
            batch_size_per_buffer = batch_size // num_buffers
            sample_start = (local_idx + protect_range) % local_size
            sample_len = local_size - protect_range
            sample_idx = (sample_start +
                          np.random.randint(0, sample_len, (batch_size_per_buffer, num_buffers))) % local_size
            # transpose --> (N_Buffer, N)
            sample_idx = sample_idx.transpose()
            # add frame stacking
            sample_idx = np.expand_dims(sample_idx, -1) + (np.arange(frame_stack) - (frame_stack - 1))
            # --> (N_Buffer, N_BS, N_FS)
            # circular buffer operation
            sample_idx = (sample_idx + local_size.reshape((-1, 1, 1))) % local_size.reshape((-1, 1, 1))

            # batch query preparation
            sample_idx = sample_idx.reshape(-1)
            buf_idx = np.repeat(np.arange(num_buffers), batch_size_per_buffer * frame_stack)

            # copy
            self.setstate("copy")
            for k in batch_keys:
                shared_batch.__dict__[k][:] = shared_buffer.__dict__[k][buf_idx, sample_idx] \
                    .reshape(batch_size, frame_stack, *shared_buffer.__dict__[k].shape[2:])
            # notify
            self.objects["batch"].set_ready()
