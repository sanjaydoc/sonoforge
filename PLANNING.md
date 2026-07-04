# SonoForge — Planning & Phase Tracker

Living checklist for the closed-loop DBTL build. Narrative detail and acceptance
tests live in [`docs/PLAN.md`](docs/PLAN.md); the technical design is in
[`docs/WHITEPAPER.md`](docs/WHITEPAPER.md). This file tracks *status*.

**Legend:** ✅ done · 🚧 in progress · ⬜ not started

---

## Phase 0 — Scaffold, docs & scientific plan ✅
- ✅ Repo skeleton (src layout), packaging (`pyproject.toml`), MIT license
- ✅ CI (GitHub Actions, py3.10/3.11: ruff + pytest), Dockerfile
- ✅ MkDocs-Material site: `README`, `WHITEPAPER`, `PLAN`, model + dataset cards
- ✅ `Candidate` / `PropertyRecord` schema + seeding utils
- ✅ Runnable script stubs (`download_data.py`, `run_cycle.py`) + passing test suite
- **Acceptance:** `pip install -e .` ✓ · `ruff check` ✓ · `pytest` ✓ · docs build ✓

## Phase 1 — Data layer & Candidate schema ⬜
- ⬜ Fetch GvpA/GvpC + ARG variant panels (RCSB / UniProt)
- ⬜ Featurization (sequence + structure) → `Candidate`
- ⬜ Disk-cached loaders; dedup + length filtering
- **Acceptance:** `download_data.py` populates real records; load→featurize→serialize round-trip test

## Phase 2 — Mamba/S4 protein language model ⬜
- ⬜ State-space PLM; pretrain hook + fine-tune on GV family (transfer learning)
- ⬜ Sequence-design head + property heads + embedding API
- ⬜ ESM-2 pseudo-log-likelihood fallback
- **Acceptance:** synthetic smoke-train converges; APIs well-shaped; fallback tested

## Phase 3 — SE(3) frame flow-matching generator ⬜
- ⬜ Conditional flow matching over residue frames; equivariant velocity field (PyTorch)
- ⬜ JAX/Flax reference equivariant layer
- ⬜ Sampler
- **Acceptance:** equivariance unit test passes; synthetic backbone sampling smoke test

## Phase 4 — Oracle stack (physics + immunogenicity) ⬜
- ⬜ Contrast proxy · **OpenMM collapse-pressure** · expressibility/solubility
- ⬜ **Immunogenicity** (MHC-II epitope load) constraint
- ⬜ ESMFold self-consistency (scRMSD, pLDDT); caching + fallbacks
- **Acceptance:** objective vector + feasibility flag; MD fallback; self-consistency within tolerance

## Phase 5 — Closed loop: constrained multi-objective active learning ⬜
- ⬜ DBTL orchestration
- ⬜ **Constrained qNEHVI** (immunogenicity as outcome constraint)
- ⬜ **GFlowNet/RL** proposal · **DPO** on pairwise screens
- ⬜ **UQ / calibration** reporting; baselines (random, NSGA-II, single-objective BO)
- **Acceptance:** hypervolume improves across cycles and beats random; calibration report emitted

## Phase 6 — Serve, benchmark & report ⬜
- ⬜ Typed API + **Gradio** app + Streamlit dashboard (democratization)
- ⬜ Benchmark suite (HV vs. oracle-calls; ablations)
- ⬜ Final report + figures; GitHub Pages deploy
- **Acceptance:** API + app launch; `benchmarks/` reproduces headline table; docs site publishes

## Stretch (post-v1) ⬜
- ⬜ Sonogenetic-**actuator** vertical (mechanosensitive channels; multi-state / membrane proteins)
- ⬜ Active-learning simulator harness for acquisition-strategy studies
- ⬜ Delivery/tropism objective (AAV-capsid surrogate)
