import cv2

import numpy as np

from onerl.nodes.node import Node


class VisualizerNode(Node):
    def run(self):
        # shared obs
        node_env_prefix = "{}@EnvNode.".format(self.node_ns)
        shared_obs = [v["log"].data["obs"].get() for k, v in self.global_objects.items()
                      if k.startswith(node_env_prefix)]

        # settings
        vis_delay = self.config.get("vis_delay", 1)

        # grid size (w > h, maximum h) by integer factorization
        grid_n = len(shared_obs)
        grid_h = int(np.sqrt(grid_n))
        while grid_n % grid_h:
            grid_h -= 1
        grid_w = grid_n // grid_h

        # grid
        obs_dtype = shared_obs[0].dtype
        if len(shared_obs[0].shape) == 3:
            # RGB / RG
            # N H W C
            img_h, img_w, img_c = shared_obs[0].shape
            assert img_c <= 3, "VisualizerNode: image must have <= 3 channels"
            obs_all = np.zeros((img_h * grid_h, img_w * grid_w, 3), dtype=obs_dtype)
        elif len(shared_obs[0].shape) == 2:
            # Gray
            img_h, img_w = shared_obs[0].shape
            img_c = None
            obs_all = np.zeros((img_h * grid_h, img_w * grid_w), dtype=obs_dtype)
        else:
            assert False, "VisualizerNode: Unknown observation image format"

        # window
        cv2.setNumThreads(1)
        cv2.namedWindow(self.node_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        while True:
            # copy (arrange as grid)
            self.setstate("copy")
            for y in range(grid_h):
                for x in range(grid_w):
                    idx = y * grid_w + x
                    if img_c is None:
                        obs_all[img_h * y: img_h * (y + 1), img_w * x: img_w * (x + 1)] = shared_obs[idx]
                    else:
                        obs_all[img_h * y: img_h * (y + 1), img_w * x: img_w * (x + 1), :img_c] = shared_obs[idx]

            # show
            self.setstate("show")
            cv2.imshow(self.node_name, obs_all)

            self.setstate("wait")
            cv2.waitKey(vis_delay)
