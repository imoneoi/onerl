import torch

from onerl.nodes.node import Node
from onerl.nodes.optimizer_node import OptimizerNode
from onerl.utils.dtype import numpy_to_torch_dtype_dict


class PolicyNode(Node):
    def run(self):
        # allocate device
        devices = self.config["devices"]
        device = torch.device(devices[self.node_rank % len(devices)])

        # create policy
        policy = OptimizerNode.create_algo(self.ns_config).to(device)
        policy.eval()

        policy_version = -1

        # load policy
        if "load_policy" in self.config:
            policy_state_dict = torch.load(self.config["load_policy"], map_location=device)
            policy_state_dict = {k.replace("module.", ""): v
                                 for k, v in policy_state_dict.items()}
            policy.load_state_dict(policy_state_dict, strict=False)

        # recurrent state
        is_recurrent = policy.recurrent_state() is not None

        # batch (cpu)
        batch_size = self.config["batch_size"]

        is_cpu = device.type == "cpu"
        obs_batch_cpu = torch.zeros((batch_size, self.ns_config["env"]["frame_stack"], *self.ns_config["env"]["obs_shape"]),
                                    dtype=numpy_to_torch_dtype_dict[self.ns_config["env"]["obs_dtype"]],
                                    pin_memory=not is_cpu)
        act_batch_cpu = torch.zeros((batch_size, *self.ns_config["env"]["act_shape"]),
                                    dtype=numpy_to_torch_dtype_dict[self.ns_config["env"]["act_dtype"]],
                                    pin_memory=not is_cpu)
        rstate_batch_cpu = None
        if is_recurrent:
            rstate_batch_cpu = torch.zeros((batch_size, *self.ns_config["env"]["rstate_shape"]),
                                           dtype=numpy_to_torch_dtype_dict[self.ns_config["env"]["rstate_dtype"]],
                                           pin_memory=not is_cpu)

        # batch (gpu)
        if is_cpu:
            obs_batch = obs_batch_cpu
            rstate_batch = rstate_batch_cpu
        else:
            obs_batch = torch.zeros_like(obs_batch_cpu, device=device)
            rstate_batch = torch.zeros_like(rstate_batch_cpu, device=device) if is_recurrent else None

        # shared objs
        node_env_list = self.find_all("EnvNode")
        obs_shared = {k: self.global_objects[k]["obs"].get_torch() for k in node_env_list}
        act_shared = {k: self.global_objects[k]["act"].get_torch() for k in node_env_list}
        if is_recurrent:
            rstate_shared = {k: self.global_objects[k]["rstate"].get_torch() for k in node_env_list}

        # nodes
        node_optimizer = self.find("OptimizerNode", 0, self.config.get("optimizer_namespace"))
        node_scheduler = self.find("SchedulerNode")
        node_metric = self.find("MetricNode")

        optimizer_shared = self.global_objects.get(node_optimizer)
        metric_shared = self.global_objects.get(node_metric)

        # ticking
        do_tick = self.config.get("do_tick", True)
        if (not do_tick) or (node_metric is None):
            self.log("Global step ticking disabled.")

        # policy update
        local_policy_state_dict = policy.policy_state_dict()
        shared_policy_state_dict = None

        if node_optimizer is not None:
            shared_policy_state_dict = optimizer_shared["update_state_dict"]
            shared_policy_state_dict.initialize("subscriber", device)
        else:
            self.log("OptimizerNode not found, unable to update policy.")

        # event loop
        while True:
            # fetch new version (lock free)
            if node_optimizer is not None:
                self.setstate("update_policy")
                optimizer_shared["update_lock"].acquire()
                new_version = optimizer_shared["update_version"].value
                optimizer_shared["update_lock"].release()

                if new_version > policy_version:
                    # TODO: may race condition here? (if skip 2 policy updates)
                    shared_policy_state_dict.receive(local_policy_state_dict)
                    policy_version = new_version

            # recv request
            self.setstate("wait")

            self.send(node_scheduler, self.node_name)  # clear scheduler queue
            env_queue = self.recv()

            # copy tensor
            self.setstate("copy_obs_rs")
            for idx, env_name in enumerate(env_queue):
                obs_batch_cpu[idx] = obs_shared[env_name]
            if is_recurrent:
                for idx, env_name in enumerate(env_queue):
                    rstate_batch_cpu[idx] = rstate_shared[env_name]
            # (if not cpu) to cuda
            if not is_cpu:
                obs_batch.copy_(obs_batch_cpu, non_blocking=True)
                if is_recurrent:
                    rstate_batch.copy_(rstate_batch_cpu, non_blocking=True)

            # get ticks
            self.setstate("step")
            ticks = None
            if do_tick:
                metric_shared["lock"].acquire()
                ticks = metric_shared["tick"].value  # read
                metric_shared["tick"].value = ticks + batch_size  # update
                metric_shared["lock"].release()

            # step
            with torch.no_grad():
                ret = policy(obs=obs_batch, ticks=ticks, rstate=rstate_batch)
                if is_recurrent:
                    ret_act, ret_rstate = ret
                else:
                    ret_act = ret

            # copy back
            self.setstate("copy_act_rs")
            # cuda --> pinned mem
            if not is_cpu:
                act_batch_cpu.copy_(ret_act, non_blocking=True)
                if is_recurrent:
                    rstate_batch_cpu.copy_(ret_rstate, non_blocking=True)
                torch.cuda.synchronize(device)

                ret_act = act_batch_cpu
                ret_rstate = rstate_batch_cpu

            # to env
            for idx, env_name in enumerate(env_queue):
                act_shared[env_name][...] = ret_act[idx]
            if is_recurrent:
                for idx, env_name in enumerate(env_queue):
                    rstate_shared[env_name][...] = ret_rstate[idx]
                    
            # notify
            for env_name in env_queue:
                self.send(env_name, "")
