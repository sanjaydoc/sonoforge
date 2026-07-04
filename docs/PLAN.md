# SonoForge — Build Plan & Roadmap

A phased, test-driven build of the closed-loop DBTL backbone. Each phase ships runnable code, tests, and docs, and degrades gracefully to a CPU/torch-free fallback so the loop always runs end-to-end.

## Guiding constraints

- **Reproducible:** Hydra configs, pinned deps, seeded runs, experiment tracking (MLflow), DVC-able data.
- **Tested:** every phase adds pytest coverage; CI runs the fast (non-weight) path on every push.
- **Honest:** simulated oracles are labelled as such in code and docs.
- **Laptop-friendly:** heavy modules (Mamba, flow model, OpenMM, ESMFold) each have a lightweight fallback; GPU is optional.

## Hardware profile (dev)

Tuned to run on a single modest GPU (≈6–8 GB VRAM) or CPU-only:
- small equivariant models, fp16, gradient checkpointing;
- MD restricted to top-of-Pareto candidates, cached and distilled into a surrogate;
- API-first ESMFold with a local fallback.

## Phases

### Phase 0 — Scaffold, docs & scientific plan ✅ (this commit)
Repo skeleton, packaging (`pyproject.toml`), CI, Dockerfile, MkDocs site, `README`, `WHITEPAPER`, this plan, and model/dataset **card** stubs.
**Acceptance:** `pip install -e .` succeeds; `pytest` (scaffold test) green; docs build.

### Phase 1 — Data layer & Candidate schema
GvpA/GvpC + ARG sequence/structure ingestion (RCSB/UniProt), featurization, and the typed `Candidate` schema (`contrast · collapse_pressure · expressibility · solubility · immunogenicity` + provenance). Disk-cached loaders.
**Acceptance:** `scripts/download_data.py` populates `data/`; round-trip (load → featurize → serialize) test green.

### Phase 2 — Mamba/S4 protein language model
State-space PLM: pretraining hook + **fine-tune on the GV family** (transfer learning), sequence-design head, and property heads. Fallback: ESM-2 pseudo-log-likelihood.
**Acceptance:** synthetic-data smoke train (`--synthetic`) converges; embedding + design APIs return well-shaped tensors; fallback path tested.

### Phase 3 — SE(3) frame flow-matching generator
Conditional flow matching over residue frames; equivariant velocity field (PyTorch) + **JAX/Flax** reference layer; sampler.
**Acceptance:** **equivariance test** (random `SE(3)` transform commutes) passes; synthetic backbone sampling smoke test green.

### Phase 4 — Oracle stack (incl. physics & immunogenicity)
`OracleStack` = contrast + **OpenMM collapse-pressure** + expressibility/solubility + **immunogenicity (MHC-II)**; ESMFold self-consistency (scRMSD, pLDDT). Caching + fallbacks throughout.
**Acceptance:** oracle returns objective vector + feasibility flag; MD path has elastic-network fallback; self-consistency on a known structure within tolerance.

### Phase 5 — Closed loop: constrained multi-objective active learning
DBTL orchestration + **constrained qNEHVI** (immunogenicity as outcome constraint) + **GFlowNet/RL** proposal + **DPO** on pairwise screens + **UQ/calibration** reporting. Baselines: random, NSGA-II, single-objective BO.
**Acceptance:** `run_cycle.py` improves hypervolume across cycles and beats random; calibration report emitted; constraint feasibility tracked.

### Phase 6 — Serve, benchmark & report
Typed API + **Gradio** app + Streamlit dashboard; benchmark suite (HV vs. oracle-calls, ablations); final technical report + figures; GitHub Pages deploy.
**Acceptance:** API + app launch; `benchmarks/` reproduces the headline table; docs site publishes.

## Stretch (post-v1)
- Sonogenetic-**actuator** vertical (mechanosensitive channels): multi-state / membrane-protein modeling — reuses the whole loop.
- Active-learning simulator harness to study acquisition strategies under controlled noise.
- Delivery/tropism objective tied to an AAV-capsid surrogate.

## Definition of done (v1)
A single command runs N DBTL cycles, produces a Pareto-optimal, immunogenicity-feasible design set with calibrated uncertainty, a reproducible benchmark beating baselines on sample efficiency, and a published docs site + demo app.
