"""Run N DBTL cycles of the SonoForge closed loop (Phase 5).

Seeds from the built library (or synthetic), evaluates with the OracleStack, and
optimizes with a chosen proposer (random / NSGA-II / qNEHVI), reporting the
feasible-front hypervolume trajectory.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from sonoforge.data.types import AA_ALPHABET, Candidate
from sonoforge.loop.dbtl import DBTLoop
from sonoforge.optimize import NSGA2Proposer, QNEHVIProposer, RandomProposer, botorch_available
from sonoforge.oracle import OracleStack

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load_seeds(n_seed: int, seed: int) -> list[Candidate]:
    lib = DATA_DIR / "seed_library.jsonl"
    if lib.exists():
        from sonoforge.data.dataset import load_candidates

        cands = load_candidates(lib)
        if cands:
            return cands[:n_seed]
    rng = random.Random(seed)
    return [Candidate(sequence="".join(rng.choice(AA_ALPHABET) for _ in range(60)))
            for _ in range(n_seed)]


def _proposer(name: str):
    if name == "random":
        return RandomProposer()
    if name == "nsga2":
        return NSGA2Proposer()
    if name == "qnehvi":
        if not botorch_available():
            print("botorch not installed; falling back to NSGA-II. Install '.[ml]' for qNEHVI.")
            return NSGA2Proposer()
        return QNEHVIProposer()
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SonoForge DBTL cycles.")
    ap.add_argument("--n-cycles", type=int, default=5)
    ap.add_argument("--library-size", type=int, default=16)
    ap.add_argument("--n-seed", type=int, default=16)
    ap.add_argument("--optimizer", choices=["qnehvi", "nsga2", "random"], default="nsga2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seeds = _load_seeds(args.n_seed, args.seed)
    loop = DBTLoop(OracleStack(), _proposer(args.optimizer), seed=args.seed)
    hist = loop.run(seeds, n_cycles=args.n_cycles, library_size=args.library_size)

    print(f"optimizer={args.optimizer}  seeds={len(seeds)}  cycles={args.n_cycles}")
    print("cycle  hypervolume  feasible%  library")
    rows = zip(hist.hypervolume, hist.feasible_fraction, hist.library_sizes, strict=False)
    for i, (hv, ff, ls) in enumerate(rows):
        print(f"{i:>5}  {hv:>11.4f}  {ff * 100:>8.1f}  {ls:>6}")
    print(f"final feasible Pareto set: {len(loop.pareto_candidates())} designs")


if __name__ == "__main__":
    main()
