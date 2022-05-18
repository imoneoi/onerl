from collections import deque

import numpy as np

from onerl.nodes.node import Node


class SchedulerNode(Node):
    @staticmethod
    def node_import_peer_objects(node_class: str, num: int, ns_config: dict, ns_objects: dict, all_ns_objects: dict):
        objects = Node.node_import_peer_objects(node_class, num, ns_config, ns_objects, all_ns_objects)
        for obj in objects:
            obj["policy"] = ns_objects.get("PolicyNode")

        return objects

    def run(self):
        # policy queue
        num_policy = self.node_count("PolicyNode", self.ns_config)
        policy_batch_size = self.ns_config["nodes"]["PolicyNode"]["batch_size"]

        policy_queues = [[] for _ in range(num_policy)]
        policy_queue_size = np.zeros(num_policy, dtype=np.int64)
        # env queue
        queue_env_id = deque()

        # event loop
        while True:
            self.setstate("wait")
            msgs = self.recv_all()
            # push queue
            for node_type, node_id in msgs:
                if node_type == "env":
                    # add env to wait queue
                    queue_env_id.append(node_id)
                else:
                    # clear policy queue
                    policy_queues[node_id].clear()
                    policy_queue_size[node_id] = 0

            # process queue
            self.setstate("step")
            while queue_env_id:
                # first full scheduling
                # return first non-full queue id
                policy_id = np.argmax(policy_queue_size < policy_batch_size)
                if policy_queue_size[policy_id] == policy_batch_size:
                    break

                policy_queues[policy_id].append(queue_env_id.popleft())
                policy_queue_size[policy_id] += 1

                if policy_queue_size[policy_id] == policy_batch_size:
                    self.send("policy", policy_id, policy_queues[policy_id])
