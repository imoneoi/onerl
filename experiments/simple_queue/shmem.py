import multiprocessing as mp
import numpy as np

import time


class Node:
    def __init__(self, node_name, node_config, shared_objects):
        super().__init__()
        self.node_name = node_name
        self.node_config = node_config
        self.shared_objects = shared_objects

        self.objects = self.shared_objects[self.node_name]
        self.queue = self.objects["queue"]

        self.state = None

    # Queue
    def send(self, target_name, msg):
        self.shared_objects[target_name]["queue"].put(msg)

    def recv(self):
        return self.queue.get()

    def available(self):
        return not self.queue.empty()

    # Logging
    def setstate(self, state):
        self.state = state
        self.send("Logger.0", ("state", self.node_name, time.time(), self.state))

    def log(self, key, value):
        self.send("Logger.0", ("log", self.node_name, key, value))

    # Node
    @staticmethod
    def create_shared_objects(node_config):
        return {
            "queue": mp.SimpleQueue()
        }

    def init(self):
        pass

    def run(self):
        assert False, "Node {} run not implemented".format(self.node_name)


class Logger(Node):
    def init(self):
        pass

    def run(self):
        while True:
            log_type, node_name, key, value = self.recv()

            del log_type, node_name, key, value


class Env(Node):
    @staticmethod
    def create_shared_objects(node_config):
        obj = super().create_shared_objects(node_config)
        obj.update({
            "obs": SharedArray()
        })
        return obj

    def init(self):
        self.actors = []
        for name in self.node_queues.keys():
            if name.startswith("Actor."):
                self.actors.append(name)

    def run(self):
        obs_shared = torch.zeros((1000, 1000))
        obs_shared.share_memory_()
        while True:
            # randomly send to an actor
            self.setstate("send_obs")
            actor = np.random.choice(self.actors)
            self.send(actor, (self.node_name, obs_shared))

            # recv action
            self.setstate("wait_act")
            act = self.recv()

            # step env
            self.setstate("step")
            obs_shared.copy_(obs_shared + act)

            # delete received
            del act


class Actor(Node):
    def init(self):
        torch.set_num_threads(1)

    def run(self):
        act_shared = torch.zeros((1000, 1000))
        act_shared.share_memory_()
        while True:
            # recv obs
            self.setstate("wait_obs")
            env_id, obs = self.recv()

            # calculate act
            self.setstate("calc_act")
            act_shared.copy_(torch.rand(1000, 1000))

            # send act
            self.setstate("send_act")
            self.send(env_id, act_shared)

            # delete received
            del env_id, obs


def node_worker(node_inst):
    node_inst.init()
    node_inst.run()


def launch_nodes(node_nums, config):
    # Create queues
    queues = {}
    for class_name, n in node_nums.items():
        for idx in range(n):
            node_name = "{}.{}".format(class_name, idx)
            queues[node_name] = mp.SimpleQueue()
    
    # Create nodes
    processes = []
    for class_name, n in node_nums.items():
        node_class = globals()[class_name]
        for idx in range(n):
            node_name = "{}.{}".format(class_name, idx)
            node_inst = node_class(node_name, config[class_name], queues)

            node_proc = mp.Process(target=node_worker, args=[node_inst], name=node_name)
            processes.append(node_proc)

    # Start all processes
    for proc in processes:
        proc.start()

    # Wait for processes to finish
    for proc in processes:
        proc.join()


if __name__ == "__main__":
    NODE_NUMS = {
        "Logger": 1,
        "Actor": 1,
        "Env": 1
    }
    NODE_CONFIG = {
        "Logger": {},
        "Actor": {},
        "Env": {}
    }

    launch_nodes(NODE_NUMS, NODE_CONFIG)
