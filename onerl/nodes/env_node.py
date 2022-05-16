import numpy as np

from onerl.utils.import_module import get_class_from_str
from onerl.nodes.node import Node
from onerl.utils.shared_array import SharedArray
from onerl.utils.batch.shared import BatchShared


class EnvNode(Node):
    @staticmethod
    def node_preprocess_ns_config(node_class: str, num: int, ns_config: dict):
        Node.node_preprocess_ns_config(node_class, num, ns_config)

        # create sample env
        sample_env = EnvNode.create_env(ns_config)
        # obs
        ns_config["env"]["obs_shape"] = sample_env.observation_space.shape
        ns_config["env"]["obs_dtype"] = sample_env.observation_space.dtype
        # act
        if hasattr(sample_env.action_space, "n"):
            # discrete
            ns_config["env"]["act_shape"] = ()
            ns_config["env"]["act_dtype"] = np.int64
            ns_config["env"]["act_n"] = sample_env.action_space.n
        else:
            # continuous
            ns_config["env"]["act_shape"] = sample_env.action_space.shape
            ns_config["env"]["act_dtype"] = sample_env.action_space.dtype
            ns_config["env"]["act_max"] = sample_env.action_space.high
            assert np.isclose(sample_env.action_space.low, -sample_env.action_space.high).all(), \
                "Action range must be symmetric"

        # batch
        ns_config["env"]["batch"] = {
            "obs": (ns_config["env"]["obs_shape"], ns_config["env"]["obs_dtype"]),
            "act": (ns_config["env"]["act_shape"], ns_config["env"]["act_dtype"]),
            "rew": ((), np.float32),
            "done": ((), np.float32)
        }

        # offline visualization
        if hasattr(sample_env, "load_state") and hasattr(sample_env, "save_state"):
            sample_vis_state = sample_env.save_state()
            ns_config["env"]["vis_state_shape"] = sample_vis_state.shape
            ns_config["env"]["vis_state_dtype"] = sample_vis_state.dtype

    @staticmethod
    def node_create_shared_objects(node_class: str, num: int, ns_config: dict):
        objects = Node.node_create_shared_objects(node_class, num, ns_config)
        for obj in objects:
            # env
            obj["obs"] = SharedArray((ns_config["env"]["frame_stack"], *ns_config["env"]["obs_shape"]),
                                     ns_config["env"]["obs_dtype"])
            obj["act"] = SharedArray(ns_config["env"]["act_shape"], ns_config["env"]["act_dtype"])

            # log to replay
            obj["log"] = BatchShared(ns_config["env"]["batch"], init_ready=True)

            # offline visualization
            if ("vis_state_shape" in ns_config["env"]) and (Node.node_count("VisualizerNode", ns_config) > 0):
                obj["vis_state"] = SharedArray(ns_config["env"]["vis_state_shape"], ns_config["env"]["vis_state_dtype"])

        return objects

    @staticmethod
    def create_env(ns_config: dict):
        env_config = ns_config["env"]
        env_class = get_class_from_str(env_config.get("import", ""), env_config["name"])
        env = env_class(**env_config.get("params", {}))
        return env

    def run(self):
        # acquire shared objects
        shared_obs = self.objects["obs"].get()
        shared_act = self.objects["act"].get()
        shared_log = self.objects["log"].get()

        shared_vis_state = self.objects["vis_state"].get() if "vis_state" in self.objects else None

        # find nodes
        node_scheduler = self.find("SchedulerNode")
        node_replay_buffer = self.find("ReplayBufferNode")
        if node_replay_buffer is None:
            self.log("ReplayBufferNode not found, skip storing experience.")

        # create and reset env
        env = self.create_env(self.ns_config)
        obs = env.reset()
        tot_reward = 0
        while True:
            # copy obs to shared mem
            self.setstate("copy_obs")
            shared_obs[:-1] = shared_obs[1:]
            shared_obs[-1] = obs

            # request act
            self.setstate("wait_act")
            self.send(node_scheduler, self.node_name)
            self.recv()

            # step env
            self.setstate("step")
            obs_next, rew, done, info = env.step(shared_act)
            # ignore time limit induced done
            log_done = done and (not info.get("TimeLimit.truncated", False))
            tot_reward += rew
            # log
            if node_replay_buffer is not None:
                self.setstate("wait_log")
                self.objects["log"].wait_ready()

                self.setstate("copy_log")
                np.copyto(shared_log.obs, obs)
                np.copyto(shared_log.act, shared_act)
                np.copyto(shared_log.rew, rew)
                np.copyto(shared_log.done, log_done)
                self.send(node_replay_buffer, self.node_name)

            # update obs & reset
            obs = obs_next
            if done:
                self.setstate("reset")
                obs = env.reset()

                self.log_metric({"{}@episode_reward".format(self.node_ns): tot_reward})
                tot_reward = 0

            # offline visualization
            if shared_vis_state is not None:
                shared_vis_state[:] = env.save_state()
