import gym
import numpy as np
import time


class SleepEnv(gym.Wrapper):
    def __init__(self, env, dt):
        super().__init__(env)
        self.dt = dt

    def step(self, action):
        time.sleep(self.dt)
        return self.env.step(action)


def create_mujoco_env(name: str, sleep_time: float = None):
    env = gym.make(name)
    env.observation_space.dtype = np.float32  # override fp32 observation

    if sleep_time is not None:
        env = SleepEnv(env, sleep_time)
    return env
