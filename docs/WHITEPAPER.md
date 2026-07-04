# SonoForge — Technical Whitepaper

**A closed-loop DBTL optimization backbone for de novo design of ultrasound-coupling neural biomolecules**

Dr. Sanjay Anbu · Working draft

---

## 1. Motivation

Merge Labs and adjacent efforts are building **non-invasive, molecular** brain–computer interfaces: instead of implanting electrodes, they express genetically-encoded proteins that let neurons exchange information with **focused ultrasound** through intact skull and tissue. The two enabling protein classes are:

1. **Acoustic biosensors (READ).** *Gas vesicles* (GVs) are hollow, gas-filled protein nanostructures, ~50–250 nm, whose stiff shells scatter ultrasound and produce nonlinear contrast. Expressed from *acoustic reporter genes (ARGs)*, they turn gene expression and, increasingly, neural activity into an ultrasound-visible signal ([Shapiro et al., *Nat. Chem. Biol.* 2014](https://shapirolab.caltech.edu); [Farhadi et al., *Science* 2019](https://www.science.org/doi/10.1126/science.aax4804)).
2. **Sonogenetic actuators (WRITE).** Mechanosensitive membrane channels (e.g. MscL, TRP variants) gated by ultrasound pressure enable cell-type-specific neuromodulation.

The key engineered handles on a gas vesicle are set **at the level of its constituent proteins** — principally the primary shell protein **GvpA** and the scaffolding/reinforcing protein **GvpC**. Varying GvpC controls the vesicle's **collapse pressure** (the acoustic pressure at which it irreversibly buckles), which in turn sets ultrasound contrast, harmonic behaviour, and — critically — the ability to do **collapse-based acoustic multiplexing** (imaging different populations by selectively collapsing them at distinct pressures) ([Lakshmanan et al., *ACS Nano* 2016](https://pubs.acs.org/doi/10.1021/acsnano.6b03364)). Directed evolution of ARGs is an established DBTL campaign ([*ACS Synth. Biol.* 2024](https://pubs.acs.org/doi/10.1021/acssynbio.4c00283)).

**The ML problem.** Optimize protein sequences/structures to jointly (i) maximize ultrasound contrast, (ii) hit a *target* collapse pressure, (iii) remain expressible and soluble in mammalian cells, and (iv) stay below an immunogenicity ceiling — from **few, noisy, expensive** measurements. This is exactly the *"closed-loop optimization backbone"* the three Merge ML Research Scientist roles are chartered to build "from a blank slate." SonoForge is a concrete, reproducible reference for that backbone.

## 2. Design principles

- **Closed-loop first.** The unit of value is a *cycle* (design → build → test → learn), not a single model. Everything is built so that a real wet-lab data stream can replace a simulated oracle with a one-line config change.
- **Physics where it's cheap and informative.** First-principles molecular dynamics is used for the one property (collapse pressure) where mechanics dominate and a learned proxy would be data-starved — encoding a genuine first-principles prior rather than a black box.
- **Uncertainty is a first-class citizen.** Expensive oracles + tiny budgets mean the acquisition function, not the point prediction, is what matters. We track calibration explicitly.
- **Safety is a constraint, not an afterthought.** Immunogenicity enters as a hard feasibility constraint, mirroring the two dedicated immunology functions at Merge.
- **Graceful degradation.** Each heavy module has a torch-free / CPU fallback so the whole loop and the test suite run on a laptop with no weights.

## 3. System architecture

```
             ┌──────────────────────── DBTL cycle k ─────────────────────────┐
 seed lib ──▶│ DESIGN                     TEST                                │──▶ archive
             │  ├ SE(3) frame flow-        ├ ESMFold self-consistency (scRMSD) │    (Pareto front,
             │  │  matching backbones      ├ contrast proxy                    │     de-immunized,
             │  └ Mamba PLM sequences ─────├ collapse-pressure  (OpenMM MD)    │     UQ-annotated)
             │                             ├ expressibility / solubility       │
             │                             └ immunogenicity (MHC-II)  ⟂ CONSTR │
             │ LEARN                                                           │
             │  └ constrained qNEHVI  ⊕  GFlowNet/RL proposal  ⊕  DPO  ⊕  UQ   │
             └───────────────────────────────────────────────────────────────┘
```

### 3.1 Representation — Mamba/S4 protein language model (`plm/`)
A **state-space sequence model** (Mamba/S4) over amino-acid tokens, pretrained on a broad protein corpus and **transfer-learned** onto the sparse gas-vesicle / ARG family. It serves three jobs: (a) inverse-folding-style sequence design conditioned on a backbone, (b) property-prediction heads (expressibility, solubility, contrast surrogate), and (c) **learned embeddings** that become the search space for Bayesian optimization. SSMs give linear-time long-context modeling and are explicitly listed as desirable in the De Novo and Biophysics roles. *Fallback:* a profile-HMM / ESM-2 pseudo-log-likelihood scorer.

### 3.2 Geometry — SE(3) frame flow-matching generator (`generative/`)
Conditional **flow matching** over residue **frames** (translations + `SO(3)` rotations), with an equivariant network as the velocity field, generalizing the Cα-only prior work to full backbone frames (à la FrameFlow / FoldFlow). Equivariance is unit-tested. A compact **JAX/Flax** reference of the equivariant layer ships alongside the PyTorch trainer to satisfy the "Python / PyTorch / **Jax**" requirement and to serve as a clean pedagogical artifact.

### 3.3 Physics — molecular dynamics oracle (`physics/`)
An **OpenMM** pipeline estimates mechanical observables of the GV shell — a **collapse-pressure proxy** from shell rigidity / buckling response and per-variant stability deltas. This is the "first-principles prior" the Computational Biophysics role centers on. MD is expensive, so it is (a) cached, (b) invoked only on the optimizer's most promising candidates, and (c) distilled into a fast surrogate over cycles. *Fallback:* an elastic-network / sequence-based rigidity proxy.

### 3.4 Oracle stack (`oracle/`)
A combined `OracleStack` returns a vector objective plus a feasibility flag per candidate:
- **Contrast** — acoustic scattering proxy from shell geometry/stiffness (maximize).
- **Collapse pressure** — MD-derived (target a set-point; enables multiplexing).
- **Expressibility & solubility** — sequence/structure proxies (maximize).
- **Immunogenicity** — MHC-II binding / T-cell epitope load (**constraint:** stay ≤ ceiling; drives de-immunization).
All oracles are disk-cached and clearly flagged as *in-silico proxies*.

### 3.5 Learn — constrained multi-objective active learning (`optimize/`)
- **qNEHVI** (BoTorch) for batched, **noisy, constrained multi-objective** Bayesian optimization over PLM embeddings, with **outcome constraints** for immunogenicity/tolerability.
- **GFlowNet / RL** proposal to generate *diverse* high-reward sequences where the acquisition function is the reward — hitting the RL requirement and combating mode collapse.
- **DPO (direct preference optimization)** to learn from *pairwise* acoustic screens ("variant A scattered more than B"), which are cheaper and lower-variance than absolute measurements — a direct fit for "preference optimization" + "sparse, noisy, high-cost data."
- **Uncertainty quantification** via deep ensembles / GP posteriors, with a **calibration report** (coverage vs. nominal) each cycle.
- **Baselines:** random search + single-objective BO + NSGA-II, for honest benchmarking.

### 3.6 Serve — democratization (`serve/`)
A typed Python API, a **Gradio** "design-an-ARG" app, and a Streamlit dashboard (hypervolume trajectory, interactive Pareto front, calibration curves, py3Dmol structure gallery). All three JDs require "serving models to non-domain experts for democratization."

## 4. Data

- **Targets & seeds:** GvpA/GvpC sequences and available GV/ARG structures (RCSB PDB, UniProt), plus published ARG variant panels where available.
- **Pretraining corpus:** a broad protein set for the PLM; fine-tune on the GV family (transfer learning under extreme sparsity).
- **Schema:** every `Candidate` carries sequence, backbone, and a property record — `contrast`, `collapse_pressure`, `expressibility`, `solubility`, `immunogenicity`, plus provenance (cycle, parents, oracle versions). Dataset and model **cards** live in `docs/cards/`.

## 5. Evaluation

1. **Optimization quality** — hypervolume trajectory vs. random / NSGA-II / single-objective BO across cycles; feasibility rate under the immunogenicity constraint.
2. **Sample efficiency** — hypervolume vs. number of (expensive) oracle calls; the metric that matters for real wet-lab budgets.
3. **Calibration** — predicted-vs-observed coverage of the surrogate's uncertainty.
4. **Structural validity** — ESMFold self-consistency (scRMSD, pLDDT) of designed sequences against intended backbones.
5. **Ablations** — physics-prior on/off; PLM embeddings vs. one-hot for BO; DPO vs. regression targets; GFlowNet diversity vs. BO-only.

## 6. Limitations

- Oracles are **in-silico proxies**, not validated assays; absolute numbers are not experimental claims. The contribution is the *closed-loop method and engineering*, into which real data plugs directly.
- The MD collapse-pressure estimate is a **coarse-grained proxy**, not a first-principles buckling simulation of a full vesicle.
- Immunogenicity prediction is a **screening filter**, not a clinical immunogenicity assessment.
- Gas-vesicle assembly is multi-protein and cooperative; single-chain structure prediction captures only part of the biology. These are stated plainly, and every simulated signal is labelled in code.

## 7. Roadmap

See [`PLAN.md`](PLAN.md) for the phased build (0–6), hardware budget, and per-phase acceptance tests.

---

## Appendix A — Role coverage matrix

| Merge requirement | Role(s) | SonoForge component |
|---|---|---|
| SSMs / LLMs | De Novo | Mamba/S4 protein LM (`plm/`) |
| SE(3)-equivariance, flow-matching | De Novo | frame flow-matching generator (`generative/`) |
| PyTorch **+ Jax** | De Novo, Biophysics | PyTorch trainer + JAX/Flax equivariant reference |
| Molecular dynamics, first-principles priors | Biophysics | OpenMM collapse-pressure oracle (`physics/`) |
| Protein-structure modeling | Biophysics | ESMFold self-consistency (`oracle/`) |
| Probabilistic modeling, UQ | BayesOpt | ensembles/GP + calibration report (`optimize/`) |
| Representation learning | BayesOpt | PLM embeddings as BO search space |
| Bayesian optimization, active learning | BayesOpt | constrained qNEHVI loop (`optimize/`, `loop/`) |
| Preference optimization | BayesOpt | DPO on pairwise acoustic screens |
| Reinforcement learning | BayesOpt | GFlowNet/RL proposal |
| Multi-objective / constrained | all three | qNEHVI + outcome constraints |
| Transfer learning | all three | pretrain → fine-tune PLM on sparse ARG data |
| Sparse / noisy / high-cost data | all three | UQ-driven active learning + DPO + oracle caching |
| Democratization to non-experts | all three | typed API + Gradio + dashboard (`serve/`) |
| Production-grade, reproducible code | all three | typed, tested, CI, Docker, Hydra, cards |
| Domain priors / wet-lab constraints | all three | immunogenicity constraint; expressibility/delivery fields |

## Appendix B — Selected references

- Shapiro et al. Biogenic gas nanostructures as ultrasound molecular reporters. *Nat. Chem. Biol.* 2014.
- Lakshmanan et al. Molecular engineering of acoustic protein nanostructures. *ACS Nano* 2016.
- Farhadi et al. Ultrasound imaging of gene expression in mammalian cells. *Science* 2019.
- Directed evolution of acoustic reporter genes. *ACS Synth. Biol.* 2024.
- Yim et al. Fast protein backbone generation with SE(3) flow matching (FrameFlow), 2023.
- Gu & Dao. Mamba: linear-time sequence modeling with selective state spaces, 2023.
- Daulton et al. Parallel noisy-constrained EHVI (qNEHVI), *NeurIPS* 2021.
- Bengio et al. GFlowNet foundations; Jain et al. Biological sequence design with GFlowNets, 2022.
- Rafailov et al. Direct Preference Optimization, 2023.
