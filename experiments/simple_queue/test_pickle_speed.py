import pickle
import time

import torchvision
import torch


# Test CPU only
model_src = torchvision.models.resnet18()
model_dst = torchvision.models.resnet18()

N = 100
start_time = time.time()
for _ in range(N):
    s = pickle.dumps(model_src.state_dict())
    model_dst.load_state_dict(pickle.loads(s))

elapsed = (time.time() - start_time) / N
print("CPU only", elapsed)

# Test GPU
model_src = model_src.to("cuda")
model_dst = model_dst.to("cuda")

N = 100
start_time = time.time()
for _ in range(N):
    s = pickle.dumps(model_src.state_dict())
    model_dst.load_state_dict(pickle.loads(s))

elapsed = (time.time() - start_time) / N
print("GPU", elapsed)
