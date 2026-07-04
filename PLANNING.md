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

## Phase 3 — SE(3) flow-matching generator ✅
- ✅ E(3)-equivariant EGNN velocity field (PyTorch) — features invariant, coords equivariant
- ✅ Conditional (OT-path) flow matching over Cα coordinates + Euler ODE sampler
- ✅ Rigid-frame utilities (Gram–Schmidt N–Cα–C, apply/invert, proper-rotation checks)
- ✅ **JAX/Flax** reference equivariant layer (param-free + learnable Flax module)
- ✅ Verified: SE(3)-equivariance test passes; flow loss decreases; sampling smoke test
- ⬜ *(extension)* full SO(3) frame-level flow (orientation channel) on top of the Cα flow
- **Acceptance:** ✅ equivariance unit test passes (torch + jax); backbone sampling smoke test green

## Phase 4 — Oracle stack (physics + immunogenicity) ✅
- ✅ **Physics collapse pressure**: Gaussian Network Model (GNM) on backbones + β-sheet
  sequence fallback + documented **OpenMM** production hook
- ✅ Contrast (scattering) · expressibility · solubility sequence proxies
- ✅ **Immunogenicity** (MHC-II epitope-load) proxy as a hard **constraint**
- ✅ `OracleStack`: fills `PropertyRecord`, maximization-objective vector,
  collapse-closeness-to-target, feasibility flag, per-signal disk cache
- ✅ 9 new tests (38 total); end-to-end oracle demo runs
- ⬜ *(follow-up)* real ESMFold self-consistency (scRMSD/pLDDT) + full OpenMM MD
- **Acceptance:** ✅ objective vector + feasibility flag; physics has GNM+sequence fallbacks; caching verified

## Phase 5 — Closed loop: constrained multi-objective active learning ✅
- ✅ DBTL orchestration (`loop/dbtl.py`) with feasible-front hypervolume tracking
- ✅ Pareto / Monte-Carlo hypervolume utilities (maximization, deterministic)
- ✅ NSGA-II with **constrained domination** + random-search baseline (sequence GA operators)
- ✅ **Constrained qNEHVI** (BoTorch) — immunogenicity as an outcome constraint (verified running)
- ✅ **DPO / Bradley–Terry** preference optimization from pairwise screens
- ✅ **UQ**: bootstrapped ensemble surrogate + calibration report
- ✅ **GFlowNet proposer**: GRU policy trained with Trajectory Balance against a
  surrogate reward — diverse, reward-proportional library generation (verified: TB
  loss ↓, diverse valid designs, HV 0.339 → 0.370)
- ✅ Verified: NSGA-II beats random on HV (0.363 vs 0.354, same budget); qNEHVI wins overall
- ⬜ *(follow-up)* wire DPO reward + UQ acquisition into the live proposer
- **Acceptance:** ✅ HV non-decreasing & NSGA-II > random; constrained qNEHVI runs; calibration emitted

## Phase 6 — Serve, benchmark & report ✅
- ✅ Typed `SonoForgeService.design(...)` (JSON-friendly) — the democratization API
- ✅ **Gradio** app + **FastAPI** REST (`/design`, `/health`) + **Streamlit** dashboard (lazy deps)
- ✅ Benchmark suite (`sonoforge.benchmark` + `benchmarks/benchmark_optimizers.py` → CSV)
- ✅ Results report (`docs/RESULTS.md`) wired into the MkDocs nav
- ✅ Verified headline: **qNEHVI 0.439 > NSGA-II 0.389 > random 0.384** (mean HV, 5 seeds)
- ⬜ *(optional)* GitHub Pages auto-deploy workflow; py3Dmol structure gallery
- **Acceptance:** ✅ service + benchmark tested; API/app build under importorskip; report published

## Stretch (post-v1) ⬜
- ⬜ Sonogenetic-**actuator** vertical (mechanosensitive channels; multi-state / membrane proteins)
- ⬜ Active-learning simulator harness for acquisition-strategy studies
- ⬜ Delivery/tropism objective (AAV-capsid surrogate)
