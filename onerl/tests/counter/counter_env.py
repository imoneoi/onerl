from abc import ABC

import numpy as np
import gym


class CounterEnv(gym.Env, ABC):
    def __init__(self):
        super().__init__()
        self.ticks = 0
        self.need_reset = True

        self.observation_space = gym.spaces.Box(low=0, high=1000000, shape=(2,), dtype=np.int64)
        self.action_space = gym.spaces.Discrete(10)

    def reset(self):
        assert self.need_reset

        self.need_reset = False
        return np.array([self.ticks, -1], dtype=np.int64)

    def step(self, action):
        assert not self.need_reset

        self.ticks += 1
        self.need_reset = (self.ticks % 1000) == 0
        return np.array([self.ticks, action], dtype=np.int64), action, self.need_reset, {}
