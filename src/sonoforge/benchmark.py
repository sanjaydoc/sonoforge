"""Benchmark optimizers on the closed loop: mean hypervolume across seeds.

Runs each optimizer through identical DBTL budgets over several random seeds and
reports the mean final hypervolume (+ trajectory). This is the head-to-head that
substantiates "our optimizer beats the baseline" — the evidence a discovery-loop
claim needs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from sonoforge.data.types import AA_ALPHABET, Candidate
from sonoforge.loop.dbtl import DBTLoop
from sonoforge.optimize import NSGA2Proposer, QNEHVIProposer, RandomProposer, botorch_available
from sonoforge.oracle import OracleStack


@dataclass
class BenchmarkResult:
    optimizer: str
    final_hv_mean: float
    final_hv_std: float
    mean_trajectory: list[float] = field(default_factory=list)


def _proposer(name: str):
    if name == "random":
        return RandomProposer()
    if name == "qnehvi":
        return QNEHVIProposer()
    return NSGA2Proposer()


def _seeds(n: int, length: int, seed: int) -> list[Candidate]:
    rng = random.Random(seed)
    return [Candidate(sequence="".join(rng.choice(AA_ALPHABET) for _ in range(length)))
            for _ in range(n)]


def run_benchmark(
    optimizers: list[str] | None = None,
    *,
    n_seeds: int = 5,
    n_cycles: int = 6,
    library_size: int = 16,
    n_seed_lib: int = 16,
) -> list[BenchmarkResult]:
    if optimizers is None:
        optimizers = ["random", "nsga2"] + (["qnehvi"] if botorch_available() else [])
    results: list[BenchmarkResult] = []
    for name in optimizers:
        trajectories = []
        for s in range(n_seeds):
            loop = DBTLoop(OracleStack(), _proposer(name), seed=s)
            hist = loop.run(_seeds(n_seed_lib, 60, s), n_cycles=n_cycles, library_size=library_size)
            trajectories.append(hist.hypervolume)
        traj = np.array(trajectories)
        finals = traj[:, -1]
        results.append(
            BenchmarkResult(
                optimizer=name,
                final_hv_mean=float(finals.mean()),
                final_hv_std=float(finals.std()),
                mean_trajectory=traj.mean(axis=0).tolist(),
            )
        )
    return results


def format_table(results: list[BenchmarkResult]) -> str:
    lines = ["| optimizer | final HV (mean ± std) |", "|---|---|"]
    for r in sorted(results, key=lambda r: r.final_hv_mean, reverse=True):
        lines.append(f"| {r.optimizer} | {r.final_hv_mean:.4f} ± {r.final_hv_std:.4f} |")
    return "\n".join(lines)
