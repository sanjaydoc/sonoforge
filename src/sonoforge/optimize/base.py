"""Shared types for the optimizers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sonoforge.data.types import Candidate


@dataclass
class Evaluated:
    """A candidate paired with its oracle objectives and feasibility."""

    candidate: Candidate
    objectives: np.ndarray          # maximization objective vector
    feasible: bool
    violation: float = 0.0          # constraint violation (0 if feasible)


class Proposer(Protocol):
    name: str

    def propose(self, evaluated: list[Evaluated], k: int, cycle: int,
                rng: random.Random) -> list[Candidate]: ...
