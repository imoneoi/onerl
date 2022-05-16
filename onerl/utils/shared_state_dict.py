from collections import OrderedDict

import torch

from onerl.utils.dtype import torch_to_numpy_dtype_dict
from onerl.utils.shared_array import SharedArray


class SharedStateDict:
    """A shared state_dict that has one publisher and many subscribers.
    """
    def __init__(self, state_dict: OrderedDict):
        self.shared_cpu = OrderedDict((k, SharedArray(v.shape, torch_to_numpy_dtype_dict[v.dtype]))
                                       for k, v in state_dict.items())

        self.type = None
        self.device = None
        self.shared_cpu_tensor = None

    def initialize(self, type: str, device: torch.device):
        self.type = type
        self.device = device
        self.shared_cpu_tensor = OrderedDict((k, v.get_torch()) for k, v in self.shared_cpu.items())

        if device.type == "cuda":
            if type == "publisher":
                # Pin memory
                [v.pin_memory_(device) for v in self.shared_cpu.values()]
            elif type == "subscriber":
                # Create pinned memory buffer
                with torch.cuda.device(device):
                    self.local_cpu_buffer = OrderedDict((k, torch.zeros_like(v, pin_memory=True)) for k, v in self.shared_cpu_tensor.items())

    def copy_state_dict(self, src, dst):
        with torch.no_grad():
            for k, v in dst.items():
                v.copy_(src[k], non_blocking=True)

    def publish(self, state_dict):
        assert self.type == "publisher", "SharedStateDict: Must initialize as publisher"

        self.copy_state_dict(state_dict, self.shared_cpu_tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def receive(self, state_dict):
        assert self.type == "subscriber", "SharedStateDict: Must initialize as subscriber"

        if self.device.type == "cuda":
            # Shared --> pinned buffer --> device
            self.copy_state_dict(self.shared_cpu_tensor, self.local_cpu_buffer)
            self.copy_state_dict(self.local_cpu_buffer, state_dict)
        else:
            # Direct copy
            self.copy_state_dict(self.shared_cpu_tensor, state_dict)

    def __repr__(self):
        return "<SharedStateDict {}>".format(list(self.shared_cpu.keys()))
