import numpy as np

from onerl.nodes.node import Node


class SchedulerNode(Node):
    def run(self):
        # policy queue
        policy_batch_size = self.ns_config["env"]["policy_batch_size"]
        policy_queue_size = np.zeros(self.node_count("PolicyNode", self.ns_config), dtype=np.int64)
        # message queue
        msg_queue_env = []
        msg_queue_policy = []
        # prefix
        node_env_prefix = "{}@EnvNode.".format(self.node_ns)
        node_policy_prefix = "{}@PolicyNode.".format(self.node_ns)
        # event loop
        while True:
            self.setstate("wait")
            msg = self.recv()

            self.setstate("step")
            # push queue
            if msg.startswith(node_env_prefix):
                msg_queue_env.append(msg)
            elif msg.startswith(node_policy_prefix):
                msg_queue_policy.append(msg)
            else:
                assert False, "Unknown message for SchedulerNode"

            # process queue
            # Policy first, then env
            while msg_queue_policy:
                policy_id = int(msg_queue_policy.pop()[len(node_policy_prefix):])
                policy_queue_size[policy_id] = 0
            while msg_queue_env:
                # first full scheduling
                # return first non-full queue id
                target_policy_id = np.argmax(policy_queue_size < policy_batch_size)
                if policy_queue_size[target_policy_id] >= policy_batch_size:
                    break

                policy_queue_size[target_policy_id] += 1
                self.send("{}@PolicyNode.{}".format(self.node_ns, target_policy_id), msg_queue_env.pop())
