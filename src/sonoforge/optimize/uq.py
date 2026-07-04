"""Uncertainty quantification: a bootstrapped ensemble surrogate + calibration.

With expensive oracles the *uncertainty* of a surrogate matters as much as its
mean. Here a small ensemble of ridge models is fit on bootstrap resamples of the
featurized library; the spread across members is the epistemic uncertainty. The
calibration report checks whether that uncertainty is honest — i.e. whether a
nominal X% interval actually contains X% of held-out targets.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from sonoforge.data.featurize import SequenceFeaturizer


class EnsembleSurrogate:
    """Bootstrap ensemble of ridge regressors predicting a scalar objective."""

    def __init__(self, n_members: int = 12, alpha: float = 1.0, seed: int = 0) -> None:
        self.n_members = n_members
        self.alpha = alpha
        self.seed = seed
        self._featurizer = SequenceFeaturizer()
        self._members: list[Ridge] = []

    def _features(self, sequences: list[str]) -> np.ndarray:
        return self._featurizer.featurize_many(sequences)

    def fit(self, sequences: list[str], y: np.ndarray) -> EnsembleSurrogate:
        x = self._features(sequences)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.seed)
        n = len(sequences)
        self._members = []
        for _ in range(self.n_members):
            idx = rng.integers(0, n, size=n)  # bootstrap resample
            model = Ridge(alpha=self.alpha)
            model.fit(x[idx], y[idx])
            self._members.append(model)
        return self

    def predict(self, sequences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) over ensemble members."""
        x = self._features(sequences)
        preds = np.stack([m.predict(x) for m in self._members])  # (members, n)
        return preds.mean(0), preds.std(0)


def calibration_report(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray,
                       levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)) -> dict[float, float]:
    """Empirical coverage at each nominal Gaussian interval level."""
    from scipy.stats import norm

    y_true = np.asarray(y_true, dtype=float)
    std = np.clip(std, 1e-6, None)
    out: dict[float, float] = {}
    for level in levels:
        z = norm.ppf(0.5 + level / 2.0)
        inside = np.abs(y_true - mean) <= z * std
        out[level] = float(inside.mean())
    return out
