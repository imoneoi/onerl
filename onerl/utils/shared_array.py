import torch
import numpy as np

from onerl.utils.dtype import numpy_to_torch_dtype_dict


class SharedArray:
    def __init__(self, shape: tuple, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype

        self.tensor = torch.zeros(shape, dtype=numpy_to_torch_dtype_dict[dtype])
        self.tensor.share_memory_()
        assert self.tensor.is_shared()

    def pin_memory_(self, device):
        with torch.cuda.device(device):
            result = torch.cuda.cudart().cudaHostRegister(self.tensor.data_ptr(), self.tensor.numel() * self.tensor.element_size(), 0)
            assert result.value == 0, "Failed to pin memory."

        assert self.tensor.is_pinned()

    def get(self):
        return self.tensor.numpy()

    def get_torch(self):
        return self.tensor

    def __repr__(self):
        return "<SharedArray shape={}, dtype={}>".format(self.shape, self.dtype)
