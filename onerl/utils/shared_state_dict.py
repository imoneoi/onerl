from collections import OrderedDict

import torch

from onerl.utils.dtype import torch_to_numpy_dtype_dict
from onerl.utils.shared_array import SharedArray


class SharedStateDict:
    def __init__(self, state_dict):
        self.data = OrderedDict({k: SharedArray(v.shape, torch_to_numpy_dtype_dict[v.dtype])
                                 for k, v in state_dict.items()})
        self.data_tensor = None

    def start(self):
        self.data_tensor = OrderedDict({k: v.get_torch() for k, v in self.data.items()})

    def load_from(self, state_dict):
        assert self.data_tensor is not None, "SharedStateDict: Must call start() before loading"
        with torch.no_grad():
            for k, v in state_dict.items():
                self.data_tensor[k].copy_(v)

    def save_to(self, state_dict):
        assert self.data_tensor is not None, "SharedStateDict: Must call start() before saving"
        with torch.no_grad():
            for k, v in state_dict.items():
                v.copy_(self.data_tensor[k])

    def __repr__(self):
        return "<SharedStateDict {}>".format(list(self.data.keys()))
