import numpy as np

from onerl.nodes.node import Node
from onerl.utils.batch.shared import BatchShared


class SamplerNode(Node):
    @staticmethod
    def node_create_shared_objects(node_class: str, num: int, ns_config: dict):
        objects = Node.node_create_shared_objects(node_class, num, ns_config)

        # check number
        assert Node.node_count("OptimizerNode", ns_config) == num, \
               "Number of sampler nodes must equal to number of optimizer nodes."

        # batch size
        tot_batch_size = ns_config["algorithm"]["params"]["batch_size"]
        num_samplers = Node.node_count("SamplerNode", ns_config)

        assert (tot_batch_size % num_samplers) == 0, \
               "Batch size must be divisible by number of samplers."
        assert tot_batch_size >= num_samplers, \
               "Batch size must be greater than number of samplers."

        batch_size = tot_batch_size // num_samplers

        # frame stack
        frame_stack = ns_config["env"].get("sample_frame_stack")
        if frame_stack is None:
            frame_stack = ns_config["env"]["frame_stack"] + 1

        # batches
        batch_info = {
            k: ((batch_size, frame_stack, *batch_shape), batch_dtype)
            for k, (batch_shape, batch_dtype) in ns_config["env"]["batch"].items()
        }
        # recurrent
        if "rstate" in batch_info:
            # only sample initial rstate, no framestacking
            batch_info["rstate"] = ((batch_size, *ns_config["env"]["rstate_shape"]), ns_config["env"]["rstate_dtype"])

        # create batches
        for obj in objects:
            obj["batch"] = BatchShared(batch_info, init_ready=False)

        return objects

    def run(self):
        # replay buffer shared objs
        replay_buffer_objs = self.global_objects[self.find("ReplayBufferNode")]

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

        batch_size, frame_stack, *_ = list(shared_batch.__dict__.values())[0].shape

        # recurrent state
        is_recurrent = "rstate" in batch_keys
        if is_recurrent:
            batch_keys.remove("rstate")

        # lock-free params
        protect_range = self.config.get("protect_range", 10) + frame_stack + 1

        # warm start
        self.setstate("start")
        while True:
            # atomic copy idx
            shared_lock.acquire()
            local_size[...] = shared_size
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
            local_size[...] = shared_size
            local_idx[...] = shared_idx
            shared_lock.release()

            # sample (idx + protect range ... end)
            # shape (N_Buffer, N)
            batch_size_per_buffer = batch_size // num_buffers
            sample_start = (local_idx + protect_range).reshape(-1, 1)
            sample_len = (local_size - protect_range).reshape(-1, 1)
            selected_idx = sample_start + \
                np.sort(np.random.randint(0, sample_len, (num_buffers, batch_size_per_buffer)), axis=-1)
            # add frame stacking
            sample_idx = np.expand_dims(selected_idx, -1) + (np.arange(frame_stack) - (frame_stack - 1))
            # --> (N_Buffer, N_BS, N_FS)
            # circular buffer operation
            sample_idx = sample_idx % local_size.reshape(-1, 1, 1)

            # batch query preparation
            sample_idx = sample_idx.reshape(-1)
            buf_idx = np.repeat(np.arange(num_buffers), batch_size_per_buffer * frame_stack)

            # copy
            self.setstate("copy")
            for k in batch_keys:
                shared_batch.__dict__[k][...] = shared_buffer.__dict__[k][buf_idx, sample_idx] \
                    .reshape(batch_size, frame_stack, *shared_buffer.__dict__[k].shape[2:])

            if is_recurrent:
                self.setstate("copy_rs")
                # copy the rstate just before frame stacking
                rstate_sample_idx = (selected_idx - frame_stack) % local_size.reshape(-1, 1)

                rstate_sample_idx = rstate_sample_idx.reshape(-1)
                rstate_buf_idx = np.repeat(np.arange(num_buffers), batch_size_per_buffer)

                shared_batch.rstate[...] = shared_buffer.rstate[rstate_buf_idx, rstate_sample_idx] \
                    .reshape(batch_size, *shared_buffer.rstate.shape[2:])

            # notify
            self.objects["batch"].set_ready()
