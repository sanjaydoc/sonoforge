"""Download the target + seed data for SonoForge (Phase 1 will flesh this out).

Fetches gas-vesicle shell proteins (GvpA / GvpC) and available acoustic-reporter
structures/sequences from public sources (RCSB PDB, UniProt). This Phase 0 stub
documents the intended interface and writes a manifest so the rest of the loop
has a stable contract to build against.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Seed targets. UniProt accessions are placeholders to be verified in Phase 1.
SEED_TARGETS = [
    {"name": "GvpA", "role": "primary gas-vesicle shell protein", "uniprot": "TBD"},
    {"name": "GvpC", "role": "scaffolding protein; tunes collapse pressure", "uniprot": "TBD"},
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"targets": SEED_TARGETS, "note": "Phase 0 stub — Phase 1 will fetch real records."}
    out = DATA_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote seed manifest to {out}")
    print("TODO(Phase 1): fetch GvpA/GvpC records + ARG variant panels from RCSB/UniProt.")


if __name__ == "__main__":
    main()
