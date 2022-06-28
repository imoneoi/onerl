import ctypes
import torch
import numpy as np

import multiprocessing as mp
import timeit


def worker_(shared_torch, shared_np, dtype, shape):
    shared_np = np.frombuffer(shared_np, dtype=dtype).reshape(shape)
    shared_torch = shared_torch.numpy()

    src_torch = np.zeros_like(shared_torch)
    src_np = np.zeros_like(shared_np)

    def copy_torch():
        shared_torch[...] = src_torch
        src_torch[...] = shared_torch

    def copy_np():
        shared_np[...] = src_np
        src_np[...] = shared_np

    print("Torch: ", timeit.timeit(copy_torch, number=100))
    print("Numpy: ", timeit.timeit(copy_np, number=100))


shape = (3, 100, 100)
dtype_np = np.float32
dtype_torch = torch.float32

sh_torch = torch.zeros(shape, dtype=dtype_torch)
sh_torch.share_memory_()

sh_np = mp.RawArray(ctypes.c_uint8, int(np.prod(shape) * np.dtype(dtype_np).itemsize))

mp.Process(target=worker_, args=(sh_torch, sh_np, dtype_np, shape)).start()
