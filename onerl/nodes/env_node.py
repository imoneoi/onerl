import multiprocessing as mp
import importlib

import numpy as np

from onerl.nodes.node import Node
from onerl.utils.shared_array import SharedArray


class EnvNode(Node):
    @staticmethod
    def node_preprocess_global_config(global_config: dict):
        super().node_preprocess_global_config(global_config)

        # create sample env
        sample_env = EnvNode.create_env(global_config)
        # obs
        global_config["env"]["obs_shape"] = sample_env.observation_space.shape
        global_config["env"]["obs_dtype"] = sample_env.observation_space.dtype
        # act
        if hasattr(sample_env.action_space, "n"):
            # discrete
            global_config["env"]["act_shape"] = (1, )
            global_config["env"]["act_dtype"] = np.int64
            global_config["env"]["act_n"] = sample_env.action_space.n
        else:
            # continuous
            global_config["env"]["act_shape"] = sample_env.action_space.shape
            global_config["env"]["act_dtype"] = sample_env.action_space.dtype
            global_config["env"]["act_max"] = sample_env.action_space.high
            assert (sample_env.action_space.low == -sample_env.action_space.high).all(), \
                "Action range must be symmetric"

    @staticmethod
    def node_create_shared_objects(num: int, global_config: dict):
        objects = super().node_create_shared_objects(num, global_config)
        for obj in objects:
            # env
            obj["obs"] = SharedArray((global_config["env"]["frame_stack"], *global_config["env"]["obs_shape"]),
                                     global_config["env"]["obs_dtype"])
            obj["act"] = SharedArray(global_config["env"]["act_shape"], global_config["env"]["act_dtype"])

            # log to replay
            obj["log_sem"] = mp.BoundedSemaphore(1)
            obj["log_obs"] = SharedArray(global_config["env"]["obs_shape"], global_config["env"]["obs_dtype"])
            obj["log_rew"] = SharedArray((1, ), dtype=np.float32)
            obj["log_done"] = SharedArray((1, ), dtype=np.bool)
            obj["log_obs_next"] = SharedArray(global_config["env"]["obs_shape"], global_config["env"]["obs_dtype"])

        return objects

    @staticmethod
    def create_env(global_config: dict):
        env_config = global_config["env"]
        if "import" in env_config:
            module = importlib.import_module(env_config["import"])
        else:
            module = globals()
        env_class = module[env_config["name"]]
        env = env_class(*env_config.get("args", []), **env_config.get("kwargs", {}))
        return env

    def run(self):
        self.dummy_init()

        # acquire shared objects
        shared = {k: v.get() for k, v in self.objects.items()}

        # create and reset env
        env = self.create_env(self.global_config)
        obs = env.reset()
        while True:
            # copy obs to shared mem
            self.setstate("copy_obs")
            shared["obs"][:-1] = shared["obs"][1:]
            shared["obs"][-1] = obs

            # request act
            self.setstate("request_act")
            self.send(self.get_node_name("SchedulerNode", 0), self.node_name)
            self.recv()

            # step env
            self.setstate("step")
            obs_next, rew, done, _ = env.step(shared["act"])
            # log
            self.setstate("log_wait")
            shared["log_sem"].acquire()

            self.setstate("log_copy")
            shared["log_obs"][:] = obs
            shared["log_rew"][:] = rew
            shared["log_done"][:] = done
            shared["log_obs_next"][:] = obs_next
            self.send(self.get_node_name("ReplayBufferNode", 0), self.node_name)

            # update obs & reset
            obs = obs_next
            if done:
                self.setstate("reset")
                obs = env.reset()
