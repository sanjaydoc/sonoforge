# Model Card — SonoForge (v0.1, Phase 0)

> Following the spirit of Mitchell et al., *Model Cards for Model Reporting* (2019).

## Overview
SonoForge is a **closed-loop DBTL optimization system**, not a single model. It composes:

| Component | Type | Status |
|---|---|---|
| Mamba/S4 protein LM | state-space sequence model | planned (Phase 2) |
| SE(3) frame flow-matching generator | equivariant generative model | planned (Phase 3) |
| OpenMM collapse-pressure oracle | physics simulation | planned (Phase 4) |
| Immunogenicity predictor | MHC-II epitope screen | planned (Phase 4) |
| Constrained qNEHVI + GFlowNet/RL + DPO | multi-objective active learning | planned (Phase 5) |

## Intended use
Research and methods demonstration: de novo design and multi-objective optimization of
gas-vesicle acoustic-reporter proteins (and, as an extension, sonogenetic actuators).
**Not** for clinical, diagnostic, or production biomanufacturing decisions.

## Out-of-scope / limitations
- Oracles are in-silico proxies; outputs are not experimental measurements.
- Immunogenicity prediction is a screening filter, not a clinical assessment.
- Single-chain modeling under-represents cooperative multi-protein GV assembly.

## Ethical & safety considerations
Immunogenicity/tolerability is encoded as a **hard constraint** to bias designs toward
human-compatible molecules. The platform is defensive/enabling in intent; it designs
imaging and neuromodulation reporters, not pathogens or toxins.

## Reproducibility
Seeded runs, pinned dependencies, Hydra configs, CI on every push, and MLflow tracking.
