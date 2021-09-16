from onerl.algorithms.algorithm import Algorithm

from onerl.algorithms.random import RandomAlgorithm

from onerl.algorithms.ddqn import DDQNAlgorithm
from onerl.algorithms.vsn import VSNAlgorithm

from onerl.algorithms.sac import SACAlgorithm

from onerl.algorithms.hwm import HWMAlgorithm
from onerl.algorithms.drp import DRPAlgorithm
from onerl.algorithms.hwm_rnn import HWMRNNAlgorithm


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
    "HWMAlgorithm",
    "DRPAlgorithm"
    "HWMRNNAlgorithm"
]
