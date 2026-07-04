# Dataset Card — SonoForge seed data (v0.1, Phase 0)

> Following the spirit of Gebru et al., *Datasheets for Datasets* (2021).

## Motivation
Provide targets and seeds for de novo design of gas-vesicle acoustic-reporter proteins:
GvpA (primary shell protein) and GvpC (scaffolding protein that tunes collapse pressure),
plus published acoustic-reporter-gene (ARG) variant panels where available.

## Composition (planned, Phase 1)
- **Targets:** GvpA / GvpC sequences and structures (RCSB PDB, UniProt).
- **Variant panels:** ARG directed-evolution variants with measured acoustic properties, where public.
- **Pretraining corpus:** a broad protein set for the PLM (transfer learning onto the sparse GV family).

## Collection & provenance
Public databases only (RCSB PDB, UniProt) and published literature. Accessions and
licenses are recorded in `data/manifest.json` at download time.

## Preprocessing
Length filtering, deduplication by sequence, and featurization to the `Candidate` schema
(`src/sonoforge/data/types.py`). Simulated oracle labels are marked as such.

## Distribution & licensing
Source structures/sequences retain their upstream licenses. No private or personal data.
This repository ships download scripts, not bulk redistributed third-party data.

## Known limitations
Public GV/ARG measurement data are sparse and heterogeneous across labs and assays —
which is precisely why the platform is built around uncertainty-aware, sample-efficient
active learning.
