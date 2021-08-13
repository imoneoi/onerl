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
        # queue
        batch_size = self.global_config["env"]["policy_batch_size"]
        batch_cpu = torch.zeros((batch_size, self.global_config["env"]["frame_stack"],
                                 *self.global_config["env"]["obs_shape"]),
                                dtype=numpy_to_torch_dtype_dict[self.global_config["env"]["obs_dtype"]])
        batch = torch.zeros_like(batch_cpu, device=device)
        # shared objs
        obs_shared = {k: v["obs"].get_torch() for k, v in self.global_objects.items() if k.startswith("EnvNode.")}
        act_shared = {k: v["act"].get_torch() for k, v in self.global_objects.items() if k.startswith("EnvNode.")}

        metric_shared = self.global_objects[self.get_node_name("MetricNode", 0)]
        optimizer_shared = self.global_objects[self.get_node_name("OptimizerNode", 0)]

        # policy update
        local_policy_state_dict = policy.policy_state_dict()
        shared_policy_state_dict = optimizer_shared["update_state_dict"]
        shared_policy_state_dict.start()

        # event loop
        while True:
            # fetch new version (lock free)
            self.setstate("update_policy")
            optimizer_shared["update_lock"].acquire()
            new_version = optimizer_shared["update_version"].value
            optimizer_shared["update_lock"].release()

            if new_version > policy_version:
                shared_policy_state_dict.save_to(local_policy_state_dict)
                policy_version = new_version

            # recv request
            self.setstate("wait")

            env_queue = []
            self.send(self.get_node_name("SchedulerNode", 0), self.node_name)  # clear scheduler queue
            while len(env_queue) < batch_size:
                env_queue.append(self.recv())

            # copy tensor & infer
            self.setstate("copy_obs")
            for idx, env_name in enumerate(env_queue):
                batch_cpu[idx] = obs_shared[env_name]
            batch.copy_(batch_cpu)

            self.setstate("step")
            # get ticks
            metric_shared["lock"].acquire()
            ticks = metric_shared["tick"].value  # read
            metric_shared["tick"].value = ticks + batch_size  # update
            metric_shared["lock"].release()
            # step
            with torch.no_grad():
                act = policy(batch, ticks)

            # copy back
            self.setstate("copy_act")
            act = act.cpu()
            for idx, env_name in enumerate(env_queue):
                act_shared[env_name].copy_(act[idx])
                self.send(env_name, "")

    def run_dummy(self):
        assert False, "PolicyNode cannot be dummy, use RandomPolicy instead"
