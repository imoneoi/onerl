import multiprocessing as mp
import ctypes
import time
import os

import wandb

from onerl.nodes.node import Node


class MetricNode(Node):
    @staticmethod
    def node_create_shared_objects(node_class: str, num: int, ns_config: dict):
        objects = Node.node_create_shared_objects(node_class, num, ns_config)

        assert num == 1, "MetricNode: There must be only one metric node."
        objects[0].update({
            "lock": mp.Lock(),
            "tick": mp.Value(ctypes.c_int64, 0, lock=False)
        })
        return objects

    def get_run_name(self):
        env_name = self.ns_config.get("env", {}).get("name", "")
        algo_name = self.ns_config.get("algorithm", {}).get("name", "Unknown")

        return "OneRL {} {} {}".format(env_name, algo_name, time.strftime("%H:%M %m-%d %Y"))

    def run(self):
        # shared objs
        shared_lock = self.objects["lock"]
        shared_tick = self.objects["tick"]
        # update to data & ticks per second
        utd_num_updates = 0
        utd_last_ticks = 0
        utd_last_time = time.time()

        utd_log_interval = self.config.get("utd_log_interval", 1.0)

        # initialize
        wandb.init(name=self.get_run_name(), config=self.ns_config)
        # log global objects
        global_objects_log_file = os.path.join(wandb.run.dir, "global_objects.json")
        with open(global_objects_log_file, "wt") as f:
            f.write(self.global_objects.__repr__())
            f.close()
        wandb.save(global_objects_log_file)

        # event loop
        while True:
            metric = self.recv()
            # get ticks
            shared_lock.acquire()
            tick = shared_tick.value
            shared_lock.release()

            # update to data
            if "update" in metric:
                utd_num_updates += metric["update"]
                del metric["update"]

                current_time = time.time()
                if (current_time - utd_last_time >= utd_log_interval) and (tick > utd_last_ticks):
                    wandb.log({
                        "update_per_data": utd_num_updates / (tick - utd_last_ticks),
                        "ticks_per_sec": (tick - utd_last_ticks) / (current_time - utd_last_time)
                    }, step=tick)

                    utd_num_updates = 0
                    utd_last_ticks = tick
                    utd_last_time = current_time

            # log metric
            wandb.log(metric, step=tick)
