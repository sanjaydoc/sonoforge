"""Preference optimization (DPO / Bradley–Terry) from pairwise acoustic screens.

Absolute acoustic measurements are noisy and expensive; *pairwise* comparisons
("variant A scattered more than B") are cheaper and lower-variance. This module
learns a scoring function from such preferences via a Bradley–Terry logistic
model on feature differences — the same objective family as DPO — giving a
reward the proposer can rank candidates by, trained on the data wet-labs actually
produce most cheaply.
"""

from __future__ import annotations

import numpy as np

from sonoforge.data.featurize import SequenceFeaturizer


class PreferenceModel:
    """Bradley–Terry model: P(a > b) = sigma(w . (phi(a) - phi(b)))."""

    def __init__(self, lr: float = 0.1, epochs: int = 300, l2: float = 1e-3, seed: int = 0) -> None:
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self._featurizer = SequenceFeaturizer()
        self.w: np.ndarray | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, winners: list[str], losers: list[str]) -> PreferenceModel:
        xw = self._featurizer.featurize_many(winners)
        xl = self._featurizer.featurize_many(losers)
        diff = xw - xl                                  # (n, d)
        w = np.zeros(diff.shape[1])
        n = len(diff)
        for _ in range(self.epochs):
            p = self._sigmoid(diff @ w)                 # P(winner preferred)
            grad = diff.T @ (p - 1.0) / n + self.l2 * w  # maximize log-likelihood of label=1
            w -= self.lr * grad
        self.w = w
        return self

    def score(self, sequences: list[str]) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("PreferenceModel is not fitted")
        return self._featurizer.featurize_many(sequences) @ self.w


def make_preference_pairs(sequences: list[str], values: np.ndarray,
                          n_pairs: int = 200, seed: int = 0) -> tuple[list[str], list[str]]:
    """Simulate pairwise screen labels: the higher-`value` sequence is the winner."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(sequences)
    winners, losers = [], []
    for _ in range(n_pairs):
        i, j = rng.integers(0, n), rng.integers(0, n)
        if values[i] == values[j]:
            continue
        hi, lo = (i, j) if values[i] > values[j] else (j, i)
        winners.append(sequences[hi])
        losers.append(sequences[lo])
    return winners, losers
