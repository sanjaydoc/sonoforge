"""Dataset assembly for SonoForge: fetch -> filter -> dedup -> Candidates.

Turns raw :class:`SequenceRecord`s into validated :class:`Candidate`s ready for
the DBTL loop, and provides JSONL persistence so a built library is reproducible
and diff-able.
"""

from __future__ import annotations

import json
from pathlib import Path

from sonoforge.data.fetch import SequenceRecord, fetch_targets
from sonoforge.data.types import AA_ALPHABET, Candidate

_AA_SET = frozenset(AA_ALPHABET)


def sanitize(sequence: str) -> str | None:
    """Upper-case and validate; return None if it contains non-standard residues."""
    s = sequence.upper().strip()
    if not s or (set(s) - _AA_SET):
        return None
    return s


def filter_by_length(records: list[SequenceRecord], lo: int = 40, hi: int = 400) -> list[SequenceRecord]:
    return [r for r in records if lo <= len(r.sequence) <= hi]


def dedup_by_sequence(records: list[SequenceRecord]) -> list[SequenceRecord]:
    """Keep the first record for each unique (sanitized) sequence."""
    seen: set[str] = set()
    out: list[SequenceRecord] = []
    for r in records:
        s = sanitize(r.sequence)
        if s is None or s in seen:
            continue
        seen.add(s)
        out.append(r)
    return out


def records_to_candidates(records: list[SequenceRecord]) -> list[Candidate]:
    cands: list[Candidate] = []
    for r in records:
        s = sanitize(r.sequence)
        if s is None:
            continue
        cands.append(
            Candidate(
                sequence=s,
                cycle=0,
                meta={
                    "accession": r.accession,
                    "name": r.name,
                    "organism": r.organism,
                    "gene": r.gene,
                    "source": r.source,
                },
            )
        )
    return cands


def build_dataset(
    cache_dir: Path | None = None,
    *,
    lo: int = 40,
    hi: int = 400,
    allow_fallback: bool = True,
) -> list[Candidate]:
    """End-to-end: fetch targets, length-filter, dedup, convert to Candidates."""
    records = fetch_targets(cache_dir=cache_dir, allow_fallback=allow_fallback)
    records = filter_by_length(records, lo=lo, hi=hi)
    records = dedup_by_sequence(records)
    return records_to_candidates(records)


def save_candidates(candidates: list[Candidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for c in candidates:
            fh.write(json.dumps(_candidate_to_row(c)) + "\n")


def load_candidates(path: Path) -> list[Candidate]:
    out: list[Candidate] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(_row_to_candidate(json.loads(line)))
    return out


def _candidate_to_row(c: Candidate) -> dict:
    return {
        "sequence": c.sequence,
        "cycle": c.cycle,
        "parents": c.parents,
        "meta": c.meta,
        "properties": {
            "contrast": c.properties.contrast,
            "collapse_pressure": c.properties.collapse_pressure,
            "expressibility": c.properties.expressibility,
            "solubility": c.properties.solubility,
            "immunogenicity": c.properties.immunogenicity,
        },
    }


def _row_to_candidate(row: dict) -> Candidate:
    from sonoforge.data.types import PropertyRecord

    props = row.get("properties", {}) or {}
    return Candidate(
        sequence=row["sequence"],
        cycle=row.get("cycle", 0),
        parents=row.get("parents", []),
        meta=row.get("meta", {}),
        properties=PropertyRecord(**props),
    )
