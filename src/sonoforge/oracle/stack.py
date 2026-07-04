"""OracleStack — evaluate a candidate on every signal, cache, and report feasibility.

Produces a filled :class:`~sonoforge.data.types.PropertyRecord` and a
maximization-objective vector for the multi-objective optimizer, plus a
feasibility flag from the immunogenicity constraint. Collapse pressure is scored
against a *target set-point* (enabling collapse-based acoustic multiplexing), so
its objective is closeness-to-target rather than "more is better".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sonoforge.data.types import Candidate, PropertyRecord
from sonoforge.oracle import signals
from sonoforge.oracle.base import DiskCache, cache_key

# maximization objectives returned to the optimizer (immunogenicity is a constraint)
OBJECTIVE_NAMES = ("contrast", "collapse_closeness", "expressibility", "solubility")

_SIGNALS = {
    "contrast": signals.contrast,
    "collapse_pressure": signals.collapse_pressure,
    "expressibility": signals.expressibility,
    "solubility": signals.solubility,
    "immunogenicity": signals.immunogenicity,
}


def _signature(candidate: Candidate) -> str:
    h = hashlib.sha1(candidate.sequence.encode())
    if candidate.backbone is not None:
        h.update(np.asarray(candidate.backbone, dtype=float).tobytes())
    return h.hexdigest()[:16]


@dataclass
class OracleStack:
    collapse_target: float = 0.5          # MPa-scale set-point
    collapse_tolerance: float = 0.25      # width of the closeness kernel
    immunogenicity_ceiling: float = 0.25  # feasible if immunogenicity <= ceiling
    cache_dir: Path | None = None

    def _cache(self, name: str) -> DiskCache:
        path = None if self.cache_dir is None else self.cache_dir / f"{name}.json"
        return DiskCache(path)

    def evaluate(self, candidate: Candidate) -> PropertyRecord:
        sig = _signature(candidate)
        rec = candidate.properties
        for name, fn in _SIGNALS.items():
            cache = self._cache(name)
            key = cache_key(name, sig)
            val = cache.get(key)
            if val is None:
                val = fn(candidate)
                cache.put(key, val)
            setattr(rec, name, val)
        return rec

    def feasible(self, record: PropertyRecord) -> bool:
        return (record.immunogenicity or 0.0) <= self.immunogenicity_ceiling

    def collapse_closeness(self, cp: float) -> float:
        """Gaussian closeness of collapse pressure to the target set-point, in (0, 1]."""
        return float(np.exp(-((cp - self.collapse_target) ** 2) / (2 * self.collapse_tolerance ** 2)))

    def objectives(self, record: PropertyRecord) -> np.ndarray:
        """Maximization-objective vector aligned with OBJECTIVE_NAMES."""
        return np.array(
            [
                record.contrast or 0.0,
                self.collapse_closeness(record.collapse_pressure or 0.0),
                record.expressibility or 0.0,
                record.solubility or 0.0,
            ],
            dtype=float,
        )

    def evaluate_batch(self, candidates: list[Candidate]) -> list[PropertyRecord]:
        return [self.evaluate(c) for c in candidates]
