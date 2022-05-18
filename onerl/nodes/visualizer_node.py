import imp
import cv2

import numpy as np

from onerl.nodes.node import Node
from onerl.nodes.env_node import EnvNode


class VisualizerNode(Node):
    @staticmethod
    def node_import_peer_objects(node_class: str, num: int, ns_config: dict, ns_objects: dict, all_ns_objects: dict):
        objects = Node.node_import_peer_objects(node_class, num, ns_config, ns_objects, all_ns_objects)
        for obj in objects:
            obj["env"] = ns_objects.get("EnvNode")

        return objects

    def run(self):
        vis_mode = "offline" if "vis_state_shape" in self.ns_config["env"] else "obs"

        if vis_mode == "offline":
            shared_state = [env_obj["vis_state"].get() for env_obj in self.peer_objects["env"]]
            num_envs = len(shared_state)

            # create envs for render
            render_env = EnvNode.create_env(self.ns_config)

            # image shape
            img_sample = render_env.render(mode="rgb_array")
            img_w, img_h, img_c = img_sample.shape
            img_dtype = img_sample.dtype
        elif vis_mode == "obs":
            # shared obs
            shared_obs = [env_obj["obs"].get() for env_obj in self.peer_objects["env"]]
            num_envs = len(shared_obs)

            # image shape
            # FS C H W
            if len(shared_obs[0].shape) == 4:
                _, img_c, img_h, img_w = shared_obs[0].shape
                img_dtype = shared_obs[0].dtype

                assert img_c <= 3, "VisualizerNode: image must have <= 3 channels"
            else:
                assert False, "VisualizerNode: Unknown observation image format"

        # settings
        vis_delay = 1000 // self.config.get("fps", 30)

        # grid size (w > h, maximum h) by integer factorization
        grid_n = num_envs
        grid_h = int(np.sqrt(grid_n))
        while grid_n % grid_h:
            grid_h -= 1
        grid_w = grid_n // grid_h

        # RGB / RG
        full_img_c = 1 if img_c == 1 else 3
        full_img = np.zeros((img_h * grid_h, img_w * grid_w, full_img_c), dtype=img_dtype)

        # window
        cv2.setNumThreads(1)
        cv2.namedWindow(self.node_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        while True:
            # copy (arrange as grid)
            self.setstate("copy")
            for y in range(grid_h):
                for x in range(grid_w):
                    idx = y * grid_w + x

                    if vis_mode == "obs":
                        env_img = shared_obs[idx][-1].transpose((1, 2, 0))  # C H W --> H W C
                    elif vis_mode == "offline":
                        render_env.load_state(shared_state[idx])
                        env_img = render_env.render(mode="rgb_array")

                    full_img[img_h * y: img_h * (y + 1), img_w * x: img_w * (x + 1), :img_c] = env_img

            obs_all_bgr = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)

            # show
            self.setstate("show")
            cv2.imshow(self.node_name, obs_all_bgr)

            self.setstate("wait")
            cv2.waitKey(vis_delay)
