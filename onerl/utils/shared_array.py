import multiprocessing as mp
import ctypes

import torch
import numpy as np


class SharedArray:
    def __init__(self, shape: tuple, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype

        self.arr = mp.Array(ctypes.c_uint8, int(np.prod(shape) * np.dtype(dtype).itemsize), lock=False)

    def get(self):
        return np.frombuffer(self.arr, dtype=self.dtype).reshape(self.shape)

    def get_torch(self):
        # The torch Tensor and numpy array will share their underlying memory locations,
        # and changing one will change the other.
        # https://pytorch.org/tutorials/beginner/former_torchies/tensor_tutorial.html
        return torch.from_numpy(self.get())
