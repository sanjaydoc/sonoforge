"""High-level, typed service API — the model-democratization entry point.

Wraps the whole DBTL loop behind one call so a non-expert can request designs
without touching the generator/oracle/optimizer internals. Returns plain
dataclasses (JSON-friendly) so the same object backs the Gradio app, the REST
API, and notebooks.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

import numpy as np

from sonoforge.data.types import AA_ALPHABET, Candidate
from sonoforge.loop.dbtl import DBTLoop
from sonoforge.optimize import NSGA2Proposer, QNEHVIProposer, RandomProposer, botorch_available
from sonoforge.oracle import OBJECTIVE_NAMES, OracleStack


@dataclass
class DesignResult:
    sequence: str
    properties: dict[str, float]
    objectives: dict[str, float]
    feasible: bool
    score: float  # mean objective (for ranking)


@dataclass
class DesignReport:
    designs: list[DesignResult]
    hypervolume_trajectory: list[float] = field(default_factory=list)
    feasible_fraction: list[float] = field(default_factory=list)
    optimizer: str = "nsga2"
    n_evaluated: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _proposer(name: str):
    if name == "random":
        return RandomProposer()
    if name == "qnehvi" and botorch_available():
        return QNEHVIProposer()
    return NSGA2Proposer()


def _synthetic_seeds(n: int, length: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice(AA_ALPHABET) for _ in range(length)) for _ in range(n)]


class SonoForgeService:
    """One-call access to closed-loop acoustic-reporter design."""

    def __init__(self, oracle: OracleStack | None = None) -> None:
        self.oracle = oracle or OracleStack()

    def design(
        self,
        seeds: list[str] | None = None,
        *,
        n_cycles: int = 5,
        library_size: int = 16,
        optimizer: str = "nsga2",
        top_k: int = 10,
        n_seed: int = 16,
        seed: int = 0,
    ) -> DesignReport:
        seq_seeds = seeds or _synthetic_seeds(n_seed, 60, seed)
        seed_cands = [Candidate(sequence=s) for s in seq_seeds]
        loop = DBTLoop(self.oracle, _proposer(optimizer), seed=seed)
        hist = loop.run(seed_cands, n_cycles=n_cycles, library_size=library_size)

        results: list[DesignResult] = []
        for c in loop.pareto_candidates():
            rec = c.properties
            obj = self.oracle.objectives(rec)
            results.append(
                DesignResult(
                    sequence=c.sequence,
                    properties={
                        "contrast": rec.contrast,
                        "collapse_pressure": rec.collapse_pressure,
                        "expressibility": rec.expressibility,
                        "solubility": rec.solubility,
                        "immunogenicity": rec.immunogenicity,
                    },
                    objectives=dict(zip(OBJECTIVE_NAMES, obj.tolist(), strict=False)),
                    feasible=True,
                    score=float(np.mean(obj)),
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return DesignReport(
            designs=results[:top_k],
            hypervolume_trajectory=hist.hypervolume,
            feasible_fraction=hist.feasible_fraction,
            optimizer=optimizer,
            n_evaluated=len(loop.archive),
        )
