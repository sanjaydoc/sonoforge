"""GFlowNet proposer tests (torch backend; skipped when torch is absent)."""

import random

import numpy as np
import pytest

from sonoforge.data.types import AA_ALPHABET, Candidate
from sonoforge.optimize.base import Evaluated


def _evaluated(n=12, length=40, seed=0):
    from sonoforge.oracle import OracleStack

    rng = random.Random(seed)
    oracle = OracleStack()
    out = []
    for _ in range(n):
        c = Candidate(sequence="".join(rng.choice(AA_ALPHABET) for _ in range(length)))
        rec = oracle.evaluate(c)
        out.append(Evaluated(c, oracle.objectives(rec), oracle.feasible(rec)))
    return out


def test_gflownet_proposes_valid_candidates_and_learns():
    pytest.importorskip("torch")
    from sonoforge.optimize import GFlowNetProposer

    proposer = GFlowNetProposer(train_steps=40, batch=16)
    evaluated = _evaluated()
    proposed = proposer.propose(evaluated, k=8, cycle=1, rng=random.Random(0))

    assert len(proposed) == 8
    assert all(isinstance(c, Candidate) for c in proposed)
    # all generated sequences are valid and of the parent length
    assert all(set(c.sequence) <= set(AA_ALPHABET) for c in proposed)
    assert all(len(c.sequence) == 40 for c in proposed)
    # Trajectory-Balance loss should trend down over training
    assert proposer.losses[-1] < proposer.losses[0]


def test_gflownet_handles_tiny_archive():
    pytest.importorskip("torch")
    from sonoforge.optimize import GFlowNetProposer

    proposed = GFlowNetProposer().propose(_evaluated(n=2), k=3, cycle=1, rng=random.Random(0))
    assert len(proposed) == 3
    assert all(isinstance(c, Candidate) for c in proposed)


def test_gflownet_sequences_are_diverse():
    pytest.importorskip("torch")
    from sonoforge.optimize import GFlowNetProposer

    proposed = GFlowNetProposer(train_steps=30).propose(
        _evaluated(), k=8, cycle=2, rng=random.Random(1)
    )
    # GFlowNet samples proportional to reward -> should not collapse to one sequence
    assert len({c.sequence for c in proposed}) >= 2
    assert np.mean([len(c.sequence) for c in proposed]) == 40
