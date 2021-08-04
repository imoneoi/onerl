import torch

from onerl.utils.batch.shared import BatchShared


class BatchCuda:
    def __init__(self, batch_shared: BatchShared, device=None):
        self.batch_shared = batch_shared
        self.data = {k: torch.zeros(v.shape, dtype=v.dtype, device=device)
                     for k, v in self.batch_shared.data.items()}

    def copy_from(self):
        for k, v in self.data.items():
            v.copy_(self.batch_shared.data[k].get_torch())

    def wait_ready(self):
        self.batch_shared.wait_ready()
