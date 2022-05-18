import multiprocessing as mp
import ctypes

import numpy as np


class SharedArray:
    def __init__(self, shape: tuple, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype

        self.arr = mp.RawArray(ctypes.c_uint8, int(np.prod(shape) * np.dtype(dtype).itemsize))

    def get(self):
        return np.frombuffer(self.arr, dtype=self.dtype).reshape(self.shape)

    def get_torch(self, pin_memory=False, pin_to=None):
        import torch

        # The torch Tensor and numpy array will share their underlying memory locations,
        # and changing one will change the other.
        # https://pytorch.org/tutorials/beginner/former_torchies/tensor_tutorial.html
        tensor = torch.from_numpy(self.get())

        # pin memory
        if pin_memory:
            with torch.cuda.device(pin_to):
                result = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0)
                assert result.value == 0, "Failed to pin memory."

            assert tensor.is_pinned()

        return tensor

    def __repr__(self):
        return "<SharedArray shape={}, dtype={}>".format(self.shape, self.dtype)
