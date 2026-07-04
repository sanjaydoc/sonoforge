"""Constrained multi-objective Bayesian optimization via qNEHVI (BoTorch).

Fits a Gaussian-process surrogate mapping sequence features to the objective
vector **plus** the immunogenicity outcome, then maximizes the *constrained*
q-Noisy Expected Hypervolume Improvement over a pool of mutated candidates —
immunogenicity enters as a hard outcome constraint (feasible ≤ ceiling). This is
the sample-efficient optimizer the Bayesian-Optimization role centers on.

Requires ``botorch`` (install ``sonoforge[ml]``). Import is lazy so the loop and
CI fall back to NSGA-II without it.
"""

from __future__ import annotations

import random

import numpy as np

from sonoforge.data.featurize import SequenceFeaturizer
from sonoforge.data.types import Candidate
from sonoforge.optimize.base import Evaluated
from sonoforge.optimize.variation import make_child


def botorch_available() -> bool:
    try:
        import botorch  # noqa: F401
        return True
    except ImportError:
        return False


class QNEHVIProposer:
    name = "qnehvi"

    def __init__(self, immunogenicity_ceiling: float = 0.25, pool_size: int = 128,
                 mutation_rate: float = 0.06) -> None:
        self.ceiling = immunogenicity_ceiling
        self.pool_size = pool_size
        self.mutation_rate = mutation_rate
        self._featurizer = SequenceFeaturizer()

    def _candidate_pool(self, evaluated: list[Evaluated], k_parents: int, cycle: int,
                        rng: random.Random) -> list[Candidate]:
        # bias the pool toward current non-dominated / feasible members
        ranked = sorted(evaluated, key=lambda e: (not e.feasible, -float(np.sum(e.objectives))))
        top = [e.candidate for e in ranked[: max(k_parents, 4)]]
        return [make_child([rng.choice(top), rng.choice(top)], cycle, self.mutation_rate, rng)
                for _ in range(self.pool_size)]

    def propose(self, evaluated: list[Evaluated], k: int, cycle: int,
                rng: random.Random) -> list[Candidate]:
        import torch
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement,
        )
        from botorch.acquisition.multi_objective.objective import (
            IdentityMCMultiOutputObjective,
        )
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood

        if len(evaluated) < 6:  # too few points to fit a GP; defer to random mutants
            return self._candidate_pool(evaluated, 4, cycle, rng)[:k]

        seqs = [e.candidate.sequence for e in evaluated]
        x = torch.tensor(self._featurizer.featurize_many(seqs), dtype=torch.double)
        objs = np.array([e.objectives for e in evaluated], dtype=float)
        immuno = np.array([[e.violation + self.ceiling] for e in evaluated])  # reconstructed value
        m = objs.shape[1]
        y = torch.tensor(np.concatenate([objs, immuno], axis=1), dtype=torch.double)

        model = SingleTaskGP(
            x, y,
            input_transform=Normalize(d=x.shape[1]),
            outcome_transform=Standardize(m=y.shape[1]),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

        ref_point = torch.tensor(objs.min(axis=0) - 0.05, dtype=torch.double)
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=x,
            objective=IdentityMCMultiOutputObjective(outcomes=list(range(m))),
            constraints=[lambda Z: Z[..., m] - self.ceiling],  # immunogenicity <= ceiling
            prune_baseline=True,
        )

        pool = self._candidate_pool(evaluated, 6, cycle, rng)
        xpool = torch.tensor(
            self._featurizer.featurize_many([c.sequence for c in pool]), dtype=torch.double
        ).unsqueeze(1)  # (pool, q=1, d)
        with torch.no_grad():
            scores = acqf(xpool)
        order = torch.argsort(scores, descending=True).tolist()
        return [pool[i] for i in order[:k]]
