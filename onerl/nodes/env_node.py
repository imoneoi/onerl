import numpy as np

from onerl.utils.import_module import get_class_from_str
from onerl.nodes.node import Node
from onerl.utils.shared_array import SharedArray
from onerl.utils.batch.shared import BatchShared


class EnvNode(Node):
    @staticmethod
    def node_preprocess_global_config(node_class: str, num: int, global_config: dict):
        super().node_preprocess_global_config(node_class, num, global_config)

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
            assert np.isclose(sample_env.action_space.low, -sample_env.action_space.high).all(), \
                "Action range must be symmetric"

    @staticmethod
    def node_create_shared_objects(node_class: str, num: int, global_config: dict):
        objects = super().node_create_shared_objects(node_class, num, global_config)
        for obj in objects:
            # env
            obj["obs"] = SharedArray((global_config["env"]["frame_stack"], *global_config["env"]["obs_shape"]),
                                     global_config["env"]["obs_dtype"])
            obj["act"] = SharedArray(global_config["env"]["act_shape"], global_config["env"]["act_dtype"])

            # log to replay
            obj["log"] = BatchShared({
                "obs": (global_config["env"]["obs_shape"], global_config["env"]["obs_dtype"]),
                "rew": ((1, ), np.float32),
                "done": ((1, ), np.bool_),
                "obs_next": (global_config["env"]["obs_shape"], global_config["env"]["obs_dtype"])
            })

        return objects

    @staticmethod
    def create_env(global_config: dict):
        env_config = global_config["env"]
        env_class = get_class_from_str(env_config.get("import", ""), env_config["name"])
        env = env_class(**env_config.get("params", {}))
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
            self.objects["log"].wait_ready()

            self.setstate("log_copy")
            shared["log"].obs[:] = obs
            shared["log"].rew[:] = rew
            shared["log"].done[:] = done
            shared["log"].obs_next[:] = obs_next
            self.send(self.get_node_name("ReplayBufferNode", 0), self.node_name)

            # update obs & reset
            obs = obs_next
            if done:
                self.setstate("reset")
                obs = env.reset()
