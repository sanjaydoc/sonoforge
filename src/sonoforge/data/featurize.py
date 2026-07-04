"""Sequence featurization for SonoForge.

Two complementary representations:

- :class:`SequenceFeaturizer` — a fixed-length, interpretable descriptor vector
  (amino-acid composition + physicochemical summaries). Cheap, deterministic,
  numpy-only; used as the initial search space for Bayesian optimization before
  the Mamba PLM embeddings come online in Phase 2.
- :func:`one_hot` — a padded/truncated (L, 20) one-hot tensor for sequence models.

Everything here is pure/deterministic so features are reproducible across runs.
"""

from __future__ import annotations

import numpy as np

from sonoforge.data.types import AA_ALPHABET

_AA_INDEX = {aa: i for i, aa in enumerate(AA_ALPHABET)}

# Kyte & Doolittle (1982) hydropathy index.
_KD_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# Approximate net charge at physiological pH.
_CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}
_AROMATIC = frozenset("FWY")

# Reference length scale (residues) for normalizing sequence length ~ O(1).
_LEN_SCALE = 100.0


def _composition(seq: str) -> np.ndarray:
    counts = np.zeros(len(AA_ALPHABET), dtype=np.float64)
    for aa in seq:
        idx = _AA_INDEX.get(aa)
        if idx is not None:
            counts[idx] += 1.0
    total = counts.sum()
    return counts / total if total > 0 else counts


class SequenceFeaturizer:
    """Fixed-length interpretable descriptor for an amino-acid sequence."""

    #: names of the descriptor block that follows the 20 composition features
    EXTRA_FEATURES = (
        "length_norm",
        "hydropathy_mean",
        "hydropathy_std",
        "net_charge_norm",
        "aromatic_frac",
    )

    @property
    def feature_names(self) -> list[str]:
        return [f"comp_{aa}" for aa in AA_ALPHABET] + list(self.EXTRA_FEATURES)

    @property
    def n_features(self) -> int:
        return len(AA_ALPHABET) + len(self.EXTRA_FEATURES)

    def featurize(self, sequence: str) -> np.ndarray:
        seq = sequence.upper()
        comp = _composition(seq)
        n = max(len(seq), 1)
        kd = np.array([_KD_HYDROPATHY.get(aa, 0.0) for aa in seq], dtype=np.float64)
        extra = np.array(
            [
                len(seq) / _LEN_SCALE,
                float(kd.mean()) if kd.size else 0.0,
                float(kd.std()) if kd.size else 0.0,
                sum(_CHARGE.get(aa, 0.0) for aa in seq) / n,
                sum(aa in _AROMATIC for aa in seq) / n,
            ],
            dtype=np.float64,
        )
        return np.concatenate([comp, extra])

    def featurize_many(self, sequences: list[str]) -> np.ndarray:
        if not sequences:
            return np.zeros((0, self.n_features), dtype=np.float64)
        return np.stack([self.featurize(s) for s in sequences])


def one_hot(sequence: str, max_len: int) -> np.ndarray:
    """(max_len, 20) one-hot encoding, right-padded with zeros / truncated."""
    arr = np.zeros((max_len, len(AA_ALPHABET)), dtype=np.float32)
    for i, aa in enumerate(sequence.upper()[:max_len]):
        idx = _AA_INDEX.get(aa)
        if idx is not None:
            arr[i, idx] = 1.0
    return arr
