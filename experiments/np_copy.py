import numpy as np

import matplotlib.pyplot as plt
from simple_benchmark import benchmark


def copy_copyto(data):
    np.copyto(data["dst"], data["src"])

def copy_assign(data):
    data["dst"][:] = data["src"]

def copy_assign_dot(data):
    data["dst"][...] = data["src"]


data = {i: {"src": np.zeros((i, 100), dtype=np.float32), "dst": np.zeros((i, 100), dtype=np.float32)} for i in [1, 10, 100, 1000, 10000, 100000]}
b = benchmark([copy_copyto, copy_assign, copy_assign_dot], data, "array size")
b.plot()

plt.show()