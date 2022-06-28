import os
import time
import multiprocessing as mp

import gym
import numpy as np
from setproctitle import setproctitle


def worker_(env_name, rank):
    os.sched_setaffinity(0, [rank])
    setproctitle("-TestGym- Rank {}".format(rank))

    env = gym.make(env_name)
    act_space = env.action_space

    counter = 0
    last_time = time.time()

    obs = env.reset()
    while True:
        act = act_space.sample()
        obs, rew, done, info = env.step(act)

        if done:
            obs = env.reset()

        counter += 1
        if counter >= 10000:
            cur_time = time.time()

            tps = counter / (cur_time - last_time)
            print(rank, "TPS", tps)

            counter = 0
            last_time = cur_time


if __name__ == "__main__":
    env_name = "Humanoid-v3"
    tot_rank = 24

    mp.set_start_method("spawn")
    procs = [mp.Process(target=worker_, args=(env_name, rank)) for rank in range(tot_rank)]

    [p.start() for p in procs]
    [p.join() for p in procs]
