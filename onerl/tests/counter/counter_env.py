from abc import ABC

import numpy as np
import gym

from onerl.tests.test_assert import test_assert


class CounterEnv(gym.Env, ABC):
    def __init__(self):
        super().__init__()
        self.ticks = 0
        self.need_reset = True

        self.observation_space = gym.spaces.Box(low=0, high=1000000, shape=(2,), dtype=np.int64)
        self.action_space = gym.spaces.Discrete(10)

    def reset(self):
        # Assertion 1: Reset if and only if done
        test_assert(self.need_reset, "Env is reset() when not needed")

        self.need_reset = False
        return np.array([self.ticks, -1], dtype=np.int64)

    def step(self, action):
        # Assertion 1: Reset if and only if done
        test_assert(not self.need_reset, "Env is not reset after epoch done")

        self.ticks += 1
        self.need_reset = (self.ticks % 1000) == 0
        return np.array([self.ticks, action], dtype=np.int64), action, self.need_reset, {}
