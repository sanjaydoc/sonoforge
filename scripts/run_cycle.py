"""Run N DBTL cycles (Phase 0 stub).

The full closed loop lands in Phase 5. This stub wires up argument parsing and
seeding so the entrypoint contract is stable, and prints the plan.
"""

from __future__ import annotations

import argparse

from sonoforge.utils import set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SonoForge DBTL cycles.")
    ap.add_argument("--n-cycles", type=int, default=5)
    ap.add_argument("--library-size", type=int, default=32)
    ap.add_argument("--optimizer", choices=["qnehvi", "nsga2", "random"], default="qnehvi")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    print("SonoForge DBTL — Phase 0 stub")
    print(f"  cycles={args.n_cycles} library_size={args.library_size} optimizer={args.optimizer}")
    print("  DESIGN -> BUILD -> TEST -> LEARN loop lands in Phase 5 (see docs/PLAN.md).")


if __name__ == "__main__":
    main()
