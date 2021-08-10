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
        policy = OptimizerNode.create_algo(self.ns_config).to(device)
        policy.eval()

        policy_version = -1
        new_policy_state = None
        # queue
        batch_size = self.ns_config["env"]["policy_batch_size"]
        batch = torch.zeros((batch_size, self.ns_config["env"]["frame_stack"],
                             *self.ns_config["env"]["obs_shape"]),
                            dtype=numpy_to_torch_dtype_dict[self.ns_config["env"]["obs_dtype"]],
                            device=device)
        # shared objs
        obs_shared = {k: v["obs"].get_torch() for k, v in self.global_objects.items() if k.startswith("EnvNode.")}
        act_shared = {k: v["act"].get_torch() for k, v in self.global_objects.items() if k.startswith("EnvNode.")}
        # nodes
        node_optimizer = self.find("OptimizerNode", 0)
        node_scheduler = self.find("SchedulerNode")

        # event loop
        while True:
            # fetch new version
            if node_optimizer is not None:
                self.setstate("update_policy")

                self.global_objects[node_optimizer]["update_lock"].acquire()
                new_version = self.global_objects[node_optimizer]["update_version"].value
                if new_version > policy_version:
                    new_policy_state = pickle.loads(self.global_objects[node_optimizer]["update_state"])
                self.global_objects[node_optimizer]["update_lock"].release()

                if new_policy_state is not None:
                    policy.load_state_dict(new_policy_state)

                    policy_version = new_version
                    new_policy_state = None

            # recv request
            self.setstate("wait")

            env_queue = []
            self.send(node_scheduler, self.node_name)  # clear scheduler queue
            while len(env_queue) < batch_size:
                env_queue.append(self.recv())

            # copy tensor & infer
            self.setstate("copy_obs")
            for idx, env_name in enumerate(env_queue):
                batch[idx].copy_(obs_shared[env_name])

            self.setstate("step")
            with torch.no_grad():
                act = policy(batch)

            # copy back
            self.setstate("copy_act")
            for idx, env_name in enumerate(env_queue):
                act_shared[env_name].copy_(act[idx])
                self.send(env_name, "")
