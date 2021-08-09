import pickle

import torch

from onerl.nodes.node import Node
from onerl.nodes.optimizer_node import OptimizerNode
from onerl.utils.dtype import numpy_to_torch_dtype_dict


class PolicyNode(Node):
    def run(self):
        # allocate device
        devices = self.config["devices"]
        device = torch.device(devices[self.node_rank % len(devices)])
        # load policy
        policy = OptimizerNode.create_algo(self.global_config).to(device)
        policy.eval()

        policy_version = -1
        new_policy_state = None
        # queue
        batch_size = self.global_config["env"]["policy_batch_size"]
        batch = torch.zeros((batch_size, self.global_config["env"]["frame_stack"],
                             *self.global_config["env"]["obs_shape"]),
                            dtype=numpy_to_torch_dtype_dict[self.global_config["env"]["obs_dtype"]],
                            device=device)
        while True:
            # fetch new version
            self.setstate("update_policy")
            optimizer_name = self.get_node_name("OptimizerNode", 0)

            self.global_objects[optimizer_name]["update_lock"].acquire()
            new_version = self.global_objects[optimizer_name]["update_version"].value
            if new_version > policy_version:
                new_policy_state = pickle.loads(self.global_objects[optimizer_name]["update_state"])
            self.global_objects[optimizer_name]["update_lock"].release()

            if new_policy_state is not None:
                policy.load_state_dict(new_policy_state)

                policy_version = new_version
                new_policy_state = None

            # recv request
            self.setstate("wait")

            env_queue = []
            self.send(self.get_node_name("SchedulerNode", 0), self.node_name)  # clear scheduler queue
            while len(env_queue) < batch_size:
                env_queue.append(self.recv())

            # copy tensor & infer
            self.setstate("copy_obs")
            for idx, env_name in enumerate(env_queue):
                batch[idx] = self.global_objects[env_name]["obs"].get_torch()

            self.setstate("step")
            with torch.no_grad():
                act = policy(batch)

            # copy back
            self.setstate("copy_act")
            for idx, env_name in enumerate(env_queue):
                self.global_objects[env_name]["act"].get_torch().copy_(act[idx].cpu())
                self.send(env_name, "")

    def run_dummy(self):
        assert False, "PolicyNode cannot be dummy, use RandomPolicy instead"
