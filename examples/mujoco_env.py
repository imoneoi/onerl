try:
    from local_debug_logger import local_trace
except ImportError:
    def local_trace():
        pass

import gym
import numpy as np
import time

import examples.halfcheetah_v3_custom


class SleepEnv(gym.Wrapper):
    def __init__(self, env, dt):
        super().__init__(env)
        self.dt = dt

    def step(self, action):
        time.sleep(self.dt)
        return self.env.step(action)


class ActionScalingWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.scaling = np.mean(env.action_space.high)
        print("Action scaling wrapper, with factor {}".format(self.scaling))

    def action(self, action):
        return self.scaling * action

    def reverse_action(self, action):
        pass


def create_mujoco_env(name: str, sleep_time: float = None):
    env = gym.make(name)
    env.observation_space.dtype = np.float32  # override fp32 observation

    if sleep_time is not None:
        env = SleepEnv(env, sleep_time)
    if not np.isclose(env.action_space.high, 1).all():
        env = ActionScalingWrapper(env)
    return env
