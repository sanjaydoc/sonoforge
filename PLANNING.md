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

## Phase 1 — Data layer & Candidate schema ✅
- ✅ Fetch GvpA/GvpC from UniProt REST (disk-cached) + labelled synthetic offline fallback
- ✅ Sequence featurization: interpretable descriptor (`SequenceFeaturizer`) + `one_hot`
- ✅ Length filtering + case-insensitive dedup + non-standard-residue rejection
- ✅ `records_to_candidates` (provenance-carrying) + JSONL save/load round-trip
- ✅ `download_data.py` builds a `seed_library.jsonl` + manifest
- ⬜ *(deferred to Phase 3/4)* structure-based featurization (needs backbones/folding)
- **Acceptance:** ✅ 8 data tests green; offline build produces valid Candidates; round-trip verified

## Phase 2 — Mamba/S4 protein language model ✅
- ✅ Selective state-space (S6/Mamba-style) block + `ProteinSSM` (LM + property + embedding heads)
- ✅ `ProteinTokenizer` (special tokens + 20 AA), masked-LM training (`train.py`, `--synthetic`)
- ✅ Unified `make_scorer` interface with a torch-free **`ProfileScorer`** fallback (embed + pseudo-LL)
- ✅ Verified: forward/backward runs; smoke-train loss decreases (3.42 → 3.27); 8 new tests green
- ⬜ *(follow-up)* ESM-2 PLL fallback + real pretraining corpus + checkpoint save/load in loop
- **Acceptance:** ✅ synthetic smoke-train converges; APIs well-shaped; fallback tested (torch-free CI path)

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
