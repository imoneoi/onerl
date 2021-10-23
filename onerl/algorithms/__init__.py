from onerl.algorithms.algorithm import Algorithm

from onerl.algorithms.random import RandomAlgorithm

from onerl.algorithms.ddqn import DDQNAlgorithm
from onerl.algorithms.vsn import VSNAlgorithm
from onerl.algorithms.fhn import FHNAlgorithm

from onerl.algorithms.sac import SACAlgorithm
from onerl.algorithms.fhac import FHACAlgorithm

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
    "FHNAlgorithm",

    # Continuous
    "SACAlgorithm",
    "FHACAlgorithm",

    # Model-based
    "DRPAlgorithm",
    "HWMCVPAlgorithm",
    "WMZeroAlgorithm"
]
