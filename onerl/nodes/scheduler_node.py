from collections import deque

import numpy as np

from onerl.nodes.node import Node


class SchedulerNode(Node):
    def run(self):
        # policy queue
        num_policy = self.node_count("PolicyNode", self.ns_config)
        policy_batch_size = self.ns_config["nodes"]["PolicyNode"]["batch_size"]

        policy_queues = [[] for _ in range(num_policy)]
        policy_queue_size = np.zeros(num_policy, dtype=np.int64)
        # message queue
        queue_env_name = deque()

        # prefix
        node_policy_dict = {k: idx for idx, k in enumerate(self.find_all("PolicyNode"))}
        # event loop
        while True:
            self.setstate("wait")
            msgs = self.recv_all()

            self.setstate("step")
            # push queue
            for node_name in msgs:
                policy_id = node_policy_dict.get(node_name)
                if policy_id is not None:
                    # clear policy queue
                    policy_queues[policy_id].clear()
                    policy_queue_size[policy_id] = 0
                else:
                    # add env to wait queue
                    queue_env_name.append(node_name)

            # process queue
            # Policy first, then env
            while queue_env_name:
                # first full scheduling
                # return first non-full queue id
                policy_id = np.argmax(policy_queue_size < policy_batch_size)
                if policy_queue_size[policy_id] == policy_batch_size:
                    break

                policy_queues[policy_id].append(queue_env_name.popleft())
                policy_queue_size[policy_id] += 1

                if policy_queue_size[policy_id] == policy_batch_size:
                    self.send(self.node_ns + "@PolicyNode." + str(policy_id), policy_queues[policy_id])
