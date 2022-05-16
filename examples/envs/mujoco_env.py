import gym
import numpy as np
import time


class SleepWrapper(gym.Wrapper):
    def __init__(self, env, dt):
        super().__init__(env)
        self.dt = dt

    def step(self, action):
        time.sleep(self.dt)
        return self.env.step(action)


class ActionScalingWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.scaling = env.action_space.high
        self.action_space = gym.spaces.Box(-1, 1, env.action_space.shape, env.action_space.dtype)

        assert np.isclose(env.action_space.low, -env.action_space.high).all(), \
                "Action range must be symmetric"

        print("Action scaling wrapper, with factor {}".format(self.scaling))

    def action(self, action):
        return self.scaling * action

    def reverse_action(self, action):
        pass


class OfflineVisWrapper(gym.Wrapper):
    """Offline visualization wrapper, saving state of env for rendering in visualizer node."""
    def __init__(self, env):
        super().__init__(env)

    def save_state(self):
        return np.concatenate([self.unwrapped.sim.data.qpos, self.unwrapped.sim.data.qvel])

    def load_state(self, state):
        qpos_size = len(self.unwrapped.sim.data.qpos)
        self.unwrapped.set_state(state[:qpos_size], state[qpos_size:])


def create_mujoco_env(name: str, sleep_time: float = None, offline_vis: bool = True):
    env = gym.make(name)
    env.observation_space.dtype = np.float32  # override fp32 observation

    if sleep_time is not None:
        env = SleepWrapper(env, sleep_time)
    if not np.isclose(env.action_space.high, 1).all():
        env = ActionScalingWrapper(env)
    if offline_vis:
        env = OfflineVisWrapper(env)
    return env
