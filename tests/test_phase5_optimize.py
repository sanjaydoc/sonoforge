"""Phase 5 tests: Pareto/HV, NSGA-II, the DBTL loop, UQ, preference opt, qNEHVI."""

import random

import numpy as np
import pytest

from sonoforge.data.types import Candidate
from sonoforge.loop.dbtl import DBTLoop
from sonoforge.optimize import (
    EnsembleSurrogate,
    NSGA2Proposer,
    PreferenceModel,
    RandomProposer,
    calibration_report,
    hypervolume,
    make_preference_pairs,
    non_dominated_mask,
    pareto_front,
)
from sonoforge.optimize.nsga2 import fast_non_dominated_sort
from sonoforge.oracle import OracleStack


def _seeds(n=16, length=60, seed=0):
    rng = random.Random(seed)
    from sonoforge.data.types import AA_ALPHABET
    return [Candidate(sequence="".join(rng.choice(AA_ALPHABET) for _ in range(length)))
            for _ in range(n)]


# --- pareto / hypervolume --------------------------------------------------

def test_non_dominated_mask_and_front():
    objs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.2]])
    mask = non_dominated_mask(objs)
    assert mask[3] == False  # [0.2,0.2] dominated by [0.5,0.5]  # noqa: E712
    assert pareto_front(objs).shape[0] == mask.sum()


def test_hypervolume_monotone_under_domination():
    ref = np.zeros(2)
    hv_small = hypervolume(np.array([[0.5, 0.5]]), ref)
    hv_big = hypervolume(np.array([[1.0, 1.0]]), ref)
    assert hv_big > hv_small
    # adding a dominated point does not change HV (within MC tolerance)
    hv_super = hypervolume(np.array([[1.0, 1.0], [0.4, 0.4]]), ref)
    assert abs(hv_super - hv_big) < 0.02


def test_fast_non_dominated_sort_layers():
    from sonoforge.optimize.base import Evaluated
    pop = [
        Evaluated(Candidate(sequence="AAA"), np.array([1.0, 1.0]), True),
        Evaluated(Candidate(sequence="CCC"), np.array([0.5, 0.5]), True),
        Evaluated(Candidate(sequence="DDD"), np.array([0.2, 0.2]), True),
    ]
    fronts = fast_non_dominated_sort(pop)
    assert fronts[0] == [0] and fronts[1] == [1] and fronts[2] == [2]


# --- DBTL loop -------------------------------------------------------------

def test_dbtl_loop_hypervolume_non_decreasing_and_beats_seed():
    oracle = OracleStack()
    loop = DBTLoop(oracle, NSGA2Proposer(), seed=1)
    hist = loop.run(_seeds(), n_cycles=5, library_size=16)
    assert len(hist.hypervolume) == 6
    # archive only grows, so the feasible-front HV is non-decreasing
    assert all(b >= a - 1e-9 for a, b in zip(hist.hypervolume, hist.hypervolume[1:], strict=False))
    assert hist.hypervolume[-1] >= hist.hypervolume[0]
    assert len(loop.pareto_candidates()) >= 1


def test_dbtl_random_and_nsga2_both_run():
    oracle = OracleStack()
    for proposer in (RandomProposer(), NSGA2Proposer()):
        loop = DBTLoop(oracle, proposer, seed=2)
        hist = loop.run(_seeds(n=12), n_cycles=3, library_size=12)
        assert hist.hypervolume[-1] >= hist.hypervolume[0]
        assert 0.0 <= hist.feasible_fraction[-1] <= 1.0


# --- UQ + preference -------------------------------------------------------

def test_ensemble_surrogate_predicts_mean_std_and_calibrates():
    seqs = [c.sequence for c in _seeds(n=40, length=50)]
    oracle = OracleStack()
    y = np.array([oracle.evaluate(Candidate(sequence=s)).contrast for s in seqs])
    surr = EnsembleSurrogate(n_members=8).fit(seqs, y)
    mean, std = surr.predict(seqs[:10])
    assert mean.shape == (10,) and std.shape == (10,)
    assert (std >= 0).all()
    cov = calibration_report(y[:10], mean, std)
    assert all(0.0 <= v <= 1.0 for v in cov.values())


def test_preference_model_ranks_by_target():
    seqs = [c.sequence for c in _seeds(n=40, length=50)]
    oracle = OracleStack()
    vals = np.array([oracle.evaluate(Candidate(sequence=s)).contrast for s in seqs])
    winners, losers = make_preference_pairs(seqs, vals, n_pairs=300)
    pm = PreferenceModel(epochs=200).fit(winners, losers)
    scores = pm.score(seqs)
    # learned preference score should correlate positively with the true value
    corr = np.corrcoef(scores, vals)[0, 1]
    assert corr > 0.3


# --- qNEHVI (BoTorch; skipped without it) ----------------------------------

def test_qnehvi_proposer_runs():
    pytest.importorskip("botorch")
    from sonoforge.optimize import QNEHVIProposer
    from sonoforge.optimize.base import Evaluated

    oracle = OracleStack()
    evaluated = []
    for c in _seeds(n=10, length=50):
        rec = oracle.evaluate(c)
        evaluated.append(
            Evaluated(c, oracle.objectives(rec), oracle.feasible(rec),
                      max(0.0, (rec.immunogenicity or 0.0) - oracle.immunogenicity_ceiling))
        )
    proposed = QNEHVIProposer().propose(evaluated, k=4, cycle=1, rng=random.Random(0))
    assert len(proposed) == 4
    assert all(isinstance(c, Candidate) for c in proposed)
