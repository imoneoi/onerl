from collections import deque

import numpy as np

from onerl.nodes.node import Node


class SchedulerNode(Node):
    def run(self):
        # policy queue
        policy_batch_size = self.ns_config["nodes"]["PolicyNode"]["batch_size"]
        policy_queue_size = np.zeros(self.node_count("PolicyNode", self.ns_config), dtype=np.int64)
        # message queue
        queue_env_name = deque()
        queue_policy_id = []
        # prefix
        node_env_dict = {k: idx for idx, k in enumerate(self.find_all("EnvNode"))}
        node_policy_dict = {k: idx for idx, k in enumerate(self.find_all("PolicyNode"))}
        # event loop
        while True:
            self.setstate("wait")
            msg = self.recv()

            self.setstate("step")
            # push queue
            env_id = node_env_dict.get(msg, None)
            policy_id = node_policy_dict.get(msg, None)
            if env_id is not None:
                queue_env_name.append(msg)
            elif policy_id is not None:
                queue_policy_id.append(policy_id)
            else:
                assert False, "Unknown message for SchedulerNode"

            # process queue
            # Policy first, then env
            while queue_policy_id:
                policy_queue_size[queue_policy_id.pop()] = 0
            while queue_env_name:
                # first full scheduling
                # return first non-full queue id
                target_policy_id = np.argmax(policy_queue_size < policy_batch_size)
                if policy_queue_size[target_policy_id] >= policy_batch_size:
                    break

                policy_queue_size[target_policy_id] += 1
                self.send("{}@PolicyNode.{}".format(self.node_ns, target_policy_id), queue_env_name.popleft())
