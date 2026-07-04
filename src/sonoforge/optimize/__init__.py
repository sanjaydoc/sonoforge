"""Multi-objective, constrained, uncertainty-aware optimization.

- Pareto / hypervolume utilities
- NSGA-II and random-search proposers (torch-free defaults)
- qNEHVI proposer (BoTorch, lazily imported)
- ensemble UQ + calibration, and DPO-style preference optimization
"""

from sonoforge.optimize.base import Evaluated, Proposer
from sonoforge.optimize.nsga2 import NSGA2Proposer
from sonoforge.optimize.pareto import hypervolume, non_dominated_mask, pareto_front
from sonoforge.optimize.preference import PreferenceModel, make_preference_pairs
from sonoforge.optimize.qnehvi import QNEHVIProposer, botorch_available
from sonoforge.optimize.random_baseline import RandomProposer
from sonoforge.optimize.uq import EnsembleSurrogate, calibration_report

__all__ = [
    "EnsembleSurrogate",
    "Evaluated",
    "NSGA2Proposer",
    "PreferenceModel",
    "Proposer",
    "QNEHVIProposer",
    "RandomProposer",
    "botorch_available",
    "calibration_report",
    "hypervolume",
    "make_preference_pairs",
    "non_dominated_mask",
    "pareto_front",
]
