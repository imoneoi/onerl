import time

import envpool
import numpy as np
import cv2

env = envpool.make_gym("HalfCheetah-v3",

                        num_envs=1,
                        batch_size=1,

                        num_threads=0,
                        thread_affinity_offset=0)


env.async_reset()

while True:
    obs, rew, done, info = env.recv()

    if np.sum(done) != 0:
        print ("!")
        # cv2.imshow("", obs[done][0, 0])
        # cv2.waitKey(0)

    env_id = info["env_id"]

    action = np.array(
        [env.action_space.sample() for _ in range(1)]
    )
    env.send(action, env_id)

