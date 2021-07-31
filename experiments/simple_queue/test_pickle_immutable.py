import multiprocessing as mp
import pickle
import ctypes

import torch


x = torch.ones(10)
fake_x = torch.zeros(10)

arr = mp.Array(ctypes.c_uint8, len(pickle.dumps(x)), lock=False)
arr[:] = pickle.dumps(x)

x_rec = pickle.loads(arr)
print(x)
arr[:] = pickle.dumps(fake_x)
print(x_rec)
print(pickle.loads(arr))
