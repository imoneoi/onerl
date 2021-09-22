from onerl.algorithms.algorithm import Algorithm

from onerl.algorithms.random import RandomAlgorithm

from onerl.algorithms.ddqn import DDQNAlgorithm
from onerl.algorithms.vsn import VSNAlgorithm

from onerl.algorithms.sac import SACAlgorithm

from onerl.algorithms.drp import DRPAlgorithm
from onerl.algorithms.hwm_cvp import HWMCVPAlgorithm
from onerl.algorithms.wmzero import WMZeroAlgorithm

__all__ = [
    # Base
    "Algorithm",

    # For testing
    "RandomAlgorithm",

    # Discrete
    "DDQNAlgorithm",
    "VSNAlgorithm",

    # Continuous
    "SACAlgorithm",

    # Model-based
    "DRPAlgorithm",
    "HWMCVPAlgorithm",
    "WMZeroAlgorithm"
]
