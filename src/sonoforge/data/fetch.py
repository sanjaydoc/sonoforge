"""Fetch target/seed sequences for SonoForge.

Primary source is the UniProt REST API (gas-vesicle shell proteins GvpA/GvpC and
related acoustic-reporter proteins). Results are disk-cached. When the network is
unavailable (e.g. sandboxed CI), we fall back to a small set of **clearly
labelled synthetic seeds** so the pipeline still runs end-to-end — these are
marked ``source="synthetic-seed"`` and must never be mistaken for real records.
"""

from __future__ import annotations

import json
import random
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from sonoforge.data.types import AA_ALPHABET

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

# Gene names that define the flagship target family.
TARGET_GENES = ("gvpA", "gvpC")


@dataclass
class SequenceRecord:
    """A fetched (or seeded) protein record."""

    accession: str
    name: str
    sequence: str
    organism: str
    gene: str
    source: str  # "uniprot" | "synthetic-seed"

    def to_dict(self) -> dict:
        return asdict(self)


def _cache_path(cache_dir: Path, gene: str) -> Path:
    return cache_dir / f"uniprot_{gene}.json"


def fetch_uniprot(
    gene: str,
    *,
    reviewed: bool = True,
    size: int = 25,
    timeout: int = 20,
    cache_dir: Path | None = None,
) -> list[SequenceRecord]:
    """Fetch reviewed UniProt records for ``gene``. Cached; empty list on failure."""
    if cache_dir is not None:
        cached = _cache_path(cache_dir, gene)
        if cached.exists():
            raw = json.loads(cached.read_text())
            return [SequenceRecord(**r) for r in raw]

    query = f"gene:{gene}" + (" AND reviewed:true" if reviewed else "")
    url = (
        f"{UNIPROT_SEARCH}?query={urllib.parse.quote(query)}"
        f"&format=json&size={size}"
        "&fields=accession,id,sequence,organism_name,gene_primary"
    )
    records: list[SequenceRecord] = []
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (trusted host)
            payload = json.load(resp)
        for item in payload.get("results", []):
            seq = item.get("sequence", {}).get("value")
            if not seq:
                continue
            records.append(
                SequenceRecord(
                    accession=item.get("primaryAccession", "?"),
                    name=item.get("uniProtkbId", gene),
                    sequence=seq,
                    organism=item.get("organism", {}).get("scientificName", "unknown"),
                    gene=gene,
                    source="uniprot",
                )
            )
    except Exception:  # noqa: BLE001 — network/parse errors → caller falls back
        return []

    if cache_dir is not None and records:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_path(cache_dir, gene).write_text(
            json.dumps([r.to_dict() for r in records], indent=2)
        )
    return records


def synthetic_seed_records(gene: str, n: int = 8, length: int = 70, seed: int = 0) -> list[SequenceRecord]:
    """Deterministic, clearly-labelled placeholder records for offline use.

    These are random sequences of a biologically plausible length for the small,
    highly-conserved GvpA shell protein — **not** real biology. They exist so the
    loop and tests run without network access; the real fetch overwrites them.
    """
    rng = random.Random(f"{gene}:{seed}")
    out: list[SequenceRecord] = []
    for i in range(n):
        s = "".join(rng.choice(AA_ALPHABET) for _ in range(length))
        out.append(
            SequenceRecord(
                accession=f"SEED-{gene}-{i:03d}",
                name=f"{gene}_synthetic_{i:03d}",
                sequence=s,
                organism="synthetic",
                gene=gene,
                source="synthetic-seed",
            )
        )
    return out


def fetch_targets(cache_dir: Path | None = None, allow_fallback: bool = True) -> list[SequenceRecord]:
    """Fetch all target genes, falling back to synthetic seeds when offline."""
    all_records: list[SequenceRecord] = []
    for gene in TARGET_GENES:
        recs = fetch_uniprot(gene, cache_dir=cache_dir)
        if not recs and allow_fallback:
            recs = synthetic_seed_records(gene)
        all_records.extend(recs)
    return all_records
