import torch

from onerl.utils.batch.shared import BatchShared
from onerl.utils.dtype import numpy_to_torch_dtype_dict


class BatchCuda:
    def __init__(self, batch_shared: BatchShared, device=None):
        self.batch_shared = batch_shared
        self.data = {k: torch.zeros(v.shape, dtype=numpy_to_torch_dtype_dict[v.dtype], device=device)
                     for k, v in self.batch_shared.data.items()}
        # Pinned buffer for faster transfer
        # https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
        self.pinned = {k: torch.zeros(v.shape, dtype=numpy_to_torch_dtype_dict[v.dtype], pin_memory=True)
                       for k, v in self.batch_shared.data.items()}
        self.src = {k: v.get_torch() for k, v in self.batch_shared.data.items()}

    def copy_from(self):
        # CPU pageable --> CPU pinned
        for k, v in self.pinned.items():
            v.copy_(self.src[k])
        # (Async) CPU pinned --> CUDA
        for k, v in self.data.items():
            v.copy_(self.pinned[k], non_blocking=True)

    def wait_ready(self):
        self.batch_shared.wait_ready()
