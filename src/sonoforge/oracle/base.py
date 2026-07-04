"""Oracle interface + disk cache.

Every oracle maps a :class:`~sonoforge.data.types.Candidate` to a float. Oracle
outputs are **in-silico proxies** in the reference platform (clearly labelled);
real wet-lab measurements plug into the same interface. Because oracle calls are
the expensive step in a real DBTL loop, results are cached on disk keyed by a
hash of the sequence + oracle name.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from sonoforge.data.types import Candidate


class Oracle(Protocol):
    name: str

    def score(self, candidate: Candidate) -> float: ...


def cache_key(name: str, sequence: str) -> str:
    return hashlib.sha1(f"{name}:{sequence}".encode()).hexdigest()[:16]


class DiskCache:
    """Tiny JSON-file cache of {key: value} for oracle results."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._store: dict[str, float] = {}
        if path is not None and path.exists():
            self._store = json.loads(path.read_text())

    def get(self, key: str) -> float | None:
        return self._store.get(key)

    def put(self, key: str, value: float) -> None:
        self._store[key] = value
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._store))
