"""Reproduce the optimizer benchmark table + write results to CSV.

    python benchmarks/benchmark_optimizers.py --n-seeds 5 --n-cycles 6
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sonoforge.benchmark import format_table, run_benchmark

OUT = Path(__file__).resolve().parent / "results.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark SonoForge optimizers.")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-cycles", type=int, default=6)
    ap.add_argument("--library-size", type=int, default=16)
    args = ap.parse_args()

    results = run_benchmark(n_seeds=args.n_seeds, n_cycles=args.n_cycles,
                            library_size=args.library_size)
    print(format_table(results))

    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["optimizer", "final_hv_mean", "final_hv_std"])
        for r in results:
            w.writerow([r.optimizer, f"{r.final_hv_mean:.6f}", f"{r.final_hv_std:.6f}"])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
