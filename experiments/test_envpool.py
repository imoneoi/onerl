import time

import envpool
import numpy as np
import cv2

batch_size = 128
print(batch_size)

env = envpool.make_gym("Breakout-v5",
                        stack_num=1,
                        
                        num_envs=batch_size * 2,
                        batch_size=batch_size,

                        use_inter_area_resize=False,

                        img_width=88,
                        img_height=88,
                        
                        num_threads=47,
                        thread_affinity_offset=0)
action = np.array(
    [env.action_space.sample() for _ in range(batch_size)]
)

counter = 0

env.async_reset()

last_time = time.time()
while True:
    obs, rew, done, info = env.recv()

    # if np.sum(done) != 0:
    #     cv2.imshow("", obs[done][0, 0])
    #     cv2.waitKey(0)

    env_id = info["env_id"]
    env.send(action, env_id)

    counter += batch_size
    if counter >= 100000:
        cur_time = time.time()
        print("TPS", counter / (cur_time - last_time))

        counter = 0
        last_time = cur_time
