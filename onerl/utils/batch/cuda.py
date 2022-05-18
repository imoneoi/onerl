import torch

from onerl.utils.batch.shared import BatchShared
from onerl.utils.torch_dtype import numpy_to_torch_dtype_dict


class BatchCuda:
    def __init__(self, batch_shared: BatchShared, device: torch.device):
        self.batch_shared = batch_shared
        self.device = device

        # Source
        pin_memory = device.type == "cuda"
        self.src = {k: v.get_torch(pin_memory=pin_memory, pin_to=device) for k, v in self.batch_shared.data.items()}
        # CUDA buffer
        self.data = {k: torch.zeros(v.shape, dtype=numpy_to_torch_dtype_dict[v.dtype], device=device)
                     for k, v in self.batch_shared.data.items()}

    def copy_from(self):
        # (Async) CPU pinned --> CUDA
        for k, v in self.data.items():
            v.copy_(self.src[k], non_blocking=True)
        # Wait all copy ready
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def wait_ready(self):
        self.batch_shared.wait_ready()
