import torch

from onerl.utils.batch.shared import BatchShared
from onerl.utils.dtype import numpy_to_torch_dtype_dict


class BatchCuda:
    def __init__(self, batch_shared: BatchShared, device: torch.device):
        self.batch_shared = batch_shared
        self.device = device

        # Source
        self.src = {k: v.pin_memory_(device).get_torch() for k, v in self.batch_shared.data.items()}
        # CUDA buffer
        self.data = {k: torch.zeros(v.shape, dtype=numpy_to_torch_dtype_dict[v.dtype], device=device)
                     for k, v in self.batch_shared.data.items()}

    def copy_from(self):
        # (Async) CPU pinned --> CUDA
        for k, v in self.data.items():
            v.copy_(self.src[k], non_blocking=True)
        # Wait all copy ready
        torch.cuda.synchronize(self.device)

    def wait_ready(self):
        self.batch_shared.wait_ready()
