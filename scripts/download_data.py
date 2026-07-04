"""Download & assemble the SonoForge seed dataset (Phase 1).

Fetches gas-vesicle shell proteins (GvpA / GvpC) from UniProt, length-filters,
deduplicates, converts to validated ``Candidate``s, and writes a JSONL library
plus a manifest. Falls back to clearly-labelled synthetic seeds when offline so
the pipeline still runs; a warning is printed in that case.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sonoforge.data import build_dataset, save_candidates

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / "cache"
LIBRARY = DATA_DIR / "seed_library.jsonl"
MANIFEST = DATA_DIR / "manifest.json"


def main() -> None:
    candidates = build_dataset(cache_dir=CACHE_DIR)
    save_candidates(candidates, LIBRARY)

    by_source = Counter(c.meta.get("source", "unknown") for c in candidates)
    by_gene = Counter(c.meta.get("gene", "unknown") for c in candidates)
    manifest = {
        "n_candidates": len(candidates),
        "by_source": dict(by_source),
        "by_gene": dict(by_gene),
        "library": str(LIBRARY.relative_to(DATA_DIR.parent)),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    print(f"Wrote {len(candidates)} candidates -> {LIBRARY}")
    print(f"  by source: {dict(by_source)}")
    print(f"  by gene:   {dict(by_gene)}")
    if by_source.get("synthetic-seed"):
        print("  WARNING: used synthetic seeds (UniProt unreachable). "
              "Re-run with network access for real records.")


if __name__ == "__main__":
    main()
