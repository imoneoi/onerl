import numpy as np

from onerl.nodes.node import Node


class SchedulerNode(Node):
    def run(self):
        # policy queue
        policy_batch_size = self.global_config["env"]["policy_batch_size"]
        policy_queue_size = np.zeros(self.node_count_by_class("PolicyNode", self.global_config), dtype=np.int64)
        # message queue
        msg_queue_env = []
        msg_queue_policy = []
        # event loop
        while True:
            self.setstate("wait_msg")
            msg = self.recv()

            self.setstate("process_msg")
            # push queue
            if msg.startswith("EnvNode."):
                msg_queue_env.append(msg)
            elif msg.startswith("PolicyNode."):
                msg_queue_policy.append(msg)
            else:
                assert False, "Unknown message for SchedulerNode"

            # process queue
            # Policy first, then env
            while msg_queue_policy:
                policy_id = int(msg_queue_policy.pop()[len("PolicyNode."):])
                policy_queue_size[policy_id] = 0
            while msg_queue_env:
                # first full scheduling
                # return first non-full queue id
                target_policy_id = np.argmax(policy_queue_size < policy_batch_size)
                if policy_queue_size[target_policy_id] >= policy_batch_size:
                    break

                policy_queue_size[target_policy_id] += 1
                self.send("PolicyNode.{}".format(target_policy_id), msg_queue_env.pop())

    def run_dummy(self):
        assert False, "SchedulerNode cannot be dummy"
