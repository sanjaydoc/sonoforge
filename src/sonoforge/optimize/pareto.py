"""Pareto dominance, non-dominated fronts, and hypervolume (maximization).

Objectives are always framed as **maximization** here (the OracleStack emits
maximization objectives). Hypervolume is estimated by a deterministic Monte-Carlo
sampler in the unit box, which is robust for the 4-objective space used here and,
unlike a buggy exact slice, is monotonic under adding non-dominated points.
"""

from __future__ import annotations

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a dominates b (maximization): a >= b everywhere and > somewhere."""
    return bool(np.all(a >= b) and np.any(a > b))


def non_dominated_mask(objs: np.ndarray) -> np.ndarray:
    """Boolean mask of Pareto-optimal rows in objs (n, m), maximization."""
    n = objs.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i != j and keep[j] and dominates(objs[j], objs[i]):
                keep[i] = False
                break
    return keep


def pareto_front(objs: np.ndarray) -> np.ndarray:
    """Return the non-dominated objective rows."""
    if objs.size == 0:
        return objs
    return objs[non_dominated_mask(objs)]


def hypervolume(objs: np.ndarray, ref: np.ndarray | None = None,
                n_mc: int = 40000, seed: int = 0) -> float:
    """Monte-Carlo hypervolume dominated by the front above ``ref`` (maximization).

    Samples uniformly in the box [ref, upper] and returns the fraction dominated
    by any front point, times the box volume. Deterministic for a fixed seed.
    """
    if objs.size == 0:
        return 0.0
    objs = np.atleast_2d(objs)
    m = objs.shape[1]
    ref = np.zeros(m) if ref is None else np.asarray(ref, dtype=float)
    upper = objs.max(axis=0)
    span = upper - ref
    if np.any(span <= 0):
        return 0.0
    front = pareto_front(objs)
    rng = np.random.default_rng(seed)
    samples = ref + rng.random((n_mc, m)) * span
    # a sample is dominated if some front point is >= it in all objectives
    dominated = np.zeros(n_mc, dtype=bool)
    for p in front:
        dominated |= np.all(samples <= p, axis=1)
    return float(dominated.mean() * np.prod(span))
