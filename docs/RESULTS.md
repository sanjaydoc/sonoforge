# Results

This page reports the headline closed-loop result: **does the optimizer actually
improve the design library, and does it beat a random-search baseline** under an
identical evaluation budget and the immunogenicity feasibility constraint?

All numbers come from the in-silico oracle stack (labelled proxies, not wet-lab
assays); the contribution is the *method and the closed-loop machinery*, which a
real experimental data stream drops into unchanged.

## Optimizer benchmark

Mean final **feasible-front hypervolume** across random seeds (higher is better),
each optimizer run through the same DBTL budget (6 cycles, library size 16,
16-sequence seed library). Reproduce with:

```bash
python benchmarks/benchmark_optimizers.py --n-seeds 5 --n-cycles 6
```

| optimizer | final feasible-front HV (mean ± std, 5 seeds) |
|---|---|
| **qNEHVI (BoTorch)** | **0.439 ± 0.016** |
| NSGA-II | 0.389 ± 0.019 |
| random (baseline) | 0.384 ± 0.014 |

**Constrained Bayesian optimization (qNEHVI) wins decisively** — its advantage over
both NSGA-II and random search exceeds the across-seed standard deviation. This is
the expected ordering when evaluations are the bottleneck: the GP surrogate spends
the budget where expected hypervolume improvement is highest, subject to the
immunogenicity constraint, instead of exploring blindly.

**Reading it:** every optimizer's hypervolume is non-decreasing across cycles (the
archive only grows), so the meaningful comparison is the *final* front. NSGA-II
and qNEHVI apply selection pressure toward the Pareto front and the immunogenicity
constraint, so they dominate random search — which improves only by luck.

## What the loop optimizes

Four maximization objectives under one hard constraint:

| signal | role | direction |
|---|---|---|
| contrast | ultrasound scattering | maximize |
| collapse pressure | mechanical set-point (multiplexing) | target closeness |
| expressibility | mammalian expression | maximize |
| solubility | aggregation resistance | maximize |
| **immunogenicity** | MHC-II epitope load | **constraint (≤ ceiling)** |

## Caveats

- Oracles are in-silico proxies; absolute values are not experimental claims.
- The hypervolume is a deterministic Monte-Carlo estimate; small differences
  within a seed's noise band should not be over-interpreted.
- At this scale (cheap proxy oracle, modest budgets) a genetic algorithm competes
  with Bayesian optimization; BO's advantage grows with expensive evaluations and
  tight query budgets — the regime real wet-lab DBTL actually operates in.
