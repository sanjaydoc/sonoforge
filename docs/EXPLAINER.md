# What SonoForge actually does (plain English)

Designing a protein in the lab is slow and expensive, so you want a computer to
**propose the few best candidates to test next**, instead of trying thousands
blindly. SonoForge does exactly that for **gas-vesicle "acoustic reporter"
proteins** — the molecules that make brain cells visible to ultrasound (Merge
Labs' core tech).

It runs a **loop**: invent candidates → score them on 4 goals → keep the best
trade-offs → invent smarter ones → repeat. Each round is better than the last.

## The flowchart

```
        ┌──────────────────────────────────────────────────────┐
        │                                                      │
        ▼                                                      │
  ①  DESIGN                                                    │
  invent new protein sequences                                 │
  (Mamba language model / SE(3) generator / GFlowNet)          │
        │                                                      │
        ▼                                                      │
  ②  TEST  (score each candidate)                              │
  • contrast      → shows up on ultrasound?                    │
  • collapse pressure → right mechanical set-point?  (physics) │
  • expressibility / solubility → will a cell make it?         │
  • immunogenicity → SAFE? (hard limit — reject if too high)   │
        │                                                      │
        ▼                                                      │
  ③  LEARN                                                     │
  optimizer keeps the best trade-offs and                      │
  proposes a smarter next batch  ────────────── loops back ────┘
  (qNEHVI Bayesian opt / NSGA-II / GFlowNet)

        ▼  (after N rounds)
  ④  RESULTS
  a ranked shortlist of the best, safe designs
  + a graph showing it improved every round
```

## What "results" look like

Running one loop prints this:

```
cycle  hypervolume  feasible%  library
    0       0.3378      91.7      12     ← starting point
    5       0.3581      97.2      72     ← better + safer after 5 rounds
final feasible Pareto set: 31 designs
```

Higher hypervolume = better designs. It went **up every cycle** — that's the whole point.

## How to run it (on your own machine)

```bash
git clone https://github.com/sanjaydoc/sonoforge.git
cd sonoforge
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"           # core (fast, no GPU needed)

python scripts/run_cycle.py --optimizer nsga2 --n-cycles 6   # run the loop, see the table
```

That's the minimum. For the *real* ML models (Mamba, qNEHVI, GFlowNet) add:
`pip install -e ".[ml]"`.

## See it in Streamlit 📊

```bash
pip install -e ".[serve]"                          # adds streamlit, gradio, fastapi
streamlit run src/sonoforge/serve/dashboard.py
```

A browser opens with a **sidebar** (pick optimizer + cycles), a **Run** button,
and when you click it you get:

- a **hypervolume line chart** (the improvement curve),
- a **feasibility curve**, and
- a **table of the top designs** with their scores.

Prefer a click-only web app? `pip install -e ".[serve]"` then
`python -m sonoforge.serve.app` gives you the **Gradio** "design-an-ARG" app.
