"""The Design–Build–Test–Learn closed loop.

Ties the oracle stack (Test) to a proposer (Design/Learn): evaluate the seed
library, then repeatedly propose → evaluate → archive, tracking the hypervolume
of the **feasible** Pareto front each cycle. Feasibility is the immunogenicity
constraint, so the loop optimizes contrast / collapse-closeness / expressibility /
solubility *subject to* staying below the epitope-load ceiling.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from sonoforge.data.types import Candidate
from sonoforge.optimize.base import Evaluated, Proposer
from sonoforge.optimize.pareto import hypervolume, non_dominated_mask
from sonoforge.oracle import OracleStack


@dataclass
class LoopHistory:
    hypervolume: list[float] = field(default_factory=list)
    feasible_fraction: list[float] = field(default_factory=list)
    library_sizes: list[int] = field(default_factory=list)


def evaluate_candidate(candidate: Candidate, oracle: OracleStack) -> Evaluated:
    rec = oracle.evaluate(candidate)
    feasible = oracle.feasible(rec)
    violation = max(0.0, (rec.immunogenicity or 0.0) - oracle.immunogenicity_ceiling)
    return Evaluated(candidate, oracle.objectives(rec), feasible, violation)


class DBTLoop:
    def __init__(self, oracle: OracleStack, proposer: Proposer, seed: int = 0) -> None:
        self.oracle = oracle
        self.proposer = proposer
        self.rng = random.Random(seed)
        self.archive: list[Evaluated] = []

    def _feasible_front_hv(self) -> float:
        feas = [e.objectives for e in self.archive if e.feasible]
        if not feas:
            return 0.0
        return hypervolume(np.array(feas))

    def _record(self, history: LoopHistory) -> None:
        history.hypervolume.append(self._feasible_front_hv())
        n_feas = sum(e.feasible for e in self.archive)
        history.feasible_fraction.append(n_feas / max(len(self.archive), 1))
        history.library_sizes.append(len(self.archive))

    def run(self, seeds: list[Candidate], n_cycles: int = 5, library_size: int = 16) -> LoopHistory:
        self.archive = [evaluate_candidate(c, self.oracle) for c in seeds]
        history = LoopHistory()
        self._record(history)
        for cycle in range(1, n_cycles + 1):
            children = self.proposer.propose(self.archive, library_size, cycle, self.rng)
            self.archive.extend(evaluate_candidate(c, self.oracle) for c in children)
            self._record(history)
        return history

    def pareto_candidates(self) -> list[Candidate]:
        """Feasible, non-dominated candidates in the final archive."""
        feas = [e for e in self.archive if e.feasible]
        if not feas:
            return []
        mask = non_dominated_mask(np.array([e.objectives for e in feas]))
        return [feas[i].candidate for i in range(len(feas)) if mask[i]]
