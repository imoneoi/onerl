import torch

from onerl.nodes.node import Node
from onerl.nodes.optimizer_node import OptimizerNode
from onerl.utils.torch_dtype import numpy_to_torch_dtype_dict


class PolicyNode(Node):
    @staticmethod
    def node_import_peer_objects(node_class: str, num: int, ns_config: dict, ns_objects: dict, all_ns_objects: dict):
        objects = Node.node_import_peer_objects(node_class, num, ns_config, ns_objects, all_ns_objects)

        optimizer_ns = ns_config["nodes"][node_class].get("optimizer_namespace")
        for obj in objects:
            obj["env"] = ns_objects.get("EnvNode")
            obj["scheduler"] = ns_objects.get("SchedulerNode")

            if optimizer_ns is not None:
                obj["optimizer"] = all_ns_objects.get(optimizer_ns, {}).get("OptimizerNode")
            else:
                obj["optimizer"] = ns_objects.get("OptimizerNode")

        return objects

    def run(self):
        self.setup_torch_opt()  # setup torch optimizations

        # allocate device
        devices = self.config["devices"]
        device = torch.device(devices[self.node_rank % len(devices)])
        # load policy
        policy = OptimizerNode.create_algo(self.ns_config).to(device)
        policy.eval()
        if "load_policy" in self.config:
            policy_state_dict = torch.load(self.config["load_policy"], map_location=device)
            policy_state_dict = {k.replace("module.", ""): v
                                 for k, v in policy_state_dict.items()}
            policy.load_state_dict(policy_state_dict, strict=False)

        policy_version = -1
        # batch
        batch_size = self.config["batch_size"]

        is_cpu = device.type == "cpu"
        batch_cpu = torch.zeros((batch_size, self.ns_config["env"]["frame_stack"],
                                 *self.ns_config["env"]["obs_shape"]),
                                dtype=numpy_to_torch_dtype_dict[self.ns_config["env"]["obs_dtype"]],
                                pin_memory=not is_cpu)
        # batch (cpu-only mode)
        if is_cpu:
            batch = batch_cpu
        else:
            batch = torch.zeros_like(batch_cpu, device=device)
        # shared objs
        obs_shared = [env_obj["obs"].get_torch() for env_obj in self.peer_objects["env"]]
        act_shared = [env_obj["act"].get_torch() for env_obj in self.peer_objects["env"]]

        optimizer_shared = self.peer_objects["optimizer"][0] if self.has_peer("optimizer") else None
        metric_shared = self.peer_objects["metric"][0] if self.has_peer("metric") else None

        # ticking
        do_tick = self.config.get("do_tick", True)
        if (not do_tick) or (metric_shared is None):
            self.log("Global step ticking disabled.")

        # policy update
        local_policy_state_dict = policy.policy_state_dict()
        shared_policy_state_dict = None

        if optimizer_shared is not None:
            shared_policy_state_dict = optimizer_shared["update_state_dict"]
            shared_policy_state_dict.initialize("subscriber", device)
        else:
            self.log("OptimizerNode not found, unable to update policy.")

        # event loop
        while True:
            # fetch new version (lock free)
            if optimizer_shared is not None:
                self.setstate("update_policy")
                optimizer_shared["update_lock"].acquire()
                new_version = optimizer_shared["update_version"].value
                optimizer_shared["update_lock"].release()

                if new_version > policy_version:
                    # TODO: may race condition here? (if skip 2 policy updates)
                    shared_policy_state_dict.receive(local_policy_state_dict)
                    policy_version = new_version

            # request for batch
            self.setstate("wait")
            self.send("scheduler", 0, ("policy", self.node_rank))
            env_queue = self.recv()

            # copy tensor & infer
            self.setstate("copy_obs")
            for idx, env_id in enumerate(env_queue):
                batch_cpu[idx] = obs_shared[env_id]
            if not is_cpu:
                batch.copy_(batch_cpu, non_blocking=True)

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
                act = policy(batch, ticks)

            # copy back
            self.setstate("copy_act")
            if not is_cpu:
                act = act.cpu()
            for idx, env_id in enumerate(env_queue):
                act_shared[env_id].copy_(act[idx])
                self.send("env", env_id, "")
