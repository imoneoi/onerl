import multiprocessing as mp
import ctypes
import time

import wandb

from onerl.nodes.node import Node


class MetricNode(Node):
    @staticmethod
    def node_create_shared_objects(node_class: str, num: int, global_config: dict):
        objects = Node.node_create_shared_objects(node_class, num, global_config)

        assert num == 1, "MetricNode: There must be only one metric node."
        objects[0].update({
            "lock": mp.Lock(),
            "tick": mp.Value(ctypes.c_int64, 0, lock=False)
        })
        return objects

    def get_run_name(self):
        env_name = self.global_config.get("env", {}).get("name", "")
        algo_name = self.global_config.get("algorithm", {}).get("name", "Unknown")

        return "OneRL {} {} {}".format(env_name, algo_name, time.strftime("%H:%M %m-%d %Y"))

    def run(self):
        # shared objs
        shared_lock = self.objects["lock"]
        shared_tick = self.objects["tick"]
        # update to data
        num_updates = 0
        utd_log_interval = self.config.get("utd_log_interval", 1.0)
        last_utd_log = time.time()

        # initialize
        wandb.init(name=self.get_run_name(), config=self.global_config)
        while True:
            metric = self.recv()
            # get ticks
            shared_lock.acquire()
            tick = shared_tick.value
            shared_lock.release()

            # update to data
            if "update" in metric:
                num_updates += metric["update"]
                del metric["update"]

                current_time = time.time()
                if current_time - last_utd_log >= utd_log_interval:
                    last_utd_log = current_time
                    wandb.log({"update_per_data": num_updates / tick}, step=tick)

            # log metric
            wandb.log(metric, step=tick)
