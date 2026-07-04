# SonoForge — Step-by-Step Run Guide

A complete, copy-paste walkthrough to clone, install, and run SonoForge — the
closed-loop platform, the ML models, and the dashboard. Windows (cmd) is shown
first; PowerShell and macOS/Linux are at the bottom.

> **Prerequisites (one-time):**
> - **Python 3.10–3.12** (3.13/3.14 also work) — https://python.org (tick *"Add Python to PATH"*)
> - **Git** — https://git-scm.com

---

## 1. Clone and enter the folder
```cmd
git clone https://github.com/sanjaydoc/sonoforge.git
cd sonoforge
```

## 2. Create the virtual environment
```cmd
python -m venv .venv
```

## 3. Activate it (the **cmd** way)
```cmd
.venv\Scripts\activate.bat
```
✅ Your prompt should now start with **`(.venv)`** — e.g. `(.venv) C:\...\sonoforge>`

> If nothing happens or a `.ps1` opens in Notepad, you're in cmd — use `activate.bat` (above), **not** `Activate.ps1`.

## 4. Install the core (fast, no GPU)
```cmd
pip install -e ".[dev]"
```

## 5. Run the closed loop
```cmd
python scripts\run_cycle.py --optimizer nsga2 --n-cycles 6
```
You'll see the hypervolume table print out (it should climb each cycle). 🎉

You can try the other torch-free optimizer too:
```cmd
python scripts\run_cycle.py --optimizer random --n-cycles 6
```

---

## 6. See it in the dashboard (Streamlit) 📊
Install the serving extras, then launch the dashboard:
```cmd
pip install -e ".[serve]"
streamlit run src\sonoforge\serve\dashboard.py
```
Then **open http://localhost:8501 in your browser** and **leave the terminal running**.
In the page: pick an optimizer + cycles in the sidebar → click **Run design loop** →
you'll see the hypervolume chart, the feasibility curve, and a table of top designs.
Press **Ctrl+C** in the terminal when you're done.

### Prefer a click-only web app? (Gradio)
```cmd
python -m sonoforge.serve.app
```
Opens the **"design-an-ARG"** app in your browser.

### Or the REST API (FastAPI)
```cmd
python -m sonoforge.serve.api
```
Serves on http://localhost:8000 — `GET /health` and `POST /design` (JSON).

---

## 7. Use the REAL ML models (PyTorch / BoTorch) 🧠
This unlocks the Mamba protein language model, qNEHVI Bayesian optimization, and
the GFlowNet generator. It's a bigger download (~1–2 GB for PyTorch/BoTorch).
```cmd
pip install -e ".[ml]"
```
Then run the advanced optimizers:
```cmd
python scripts\run_cycle.py --optimizer qnehvi   --n-cycles 6
python scripts\run_cycle.py --optimizer gflownet --n-cycles 6
```
> qNEHVI is slower (it fits a Gaussian-process surrogate each cycle) but gives the
> best designs — it's the benchmark winner. Any `warnings.warn(...)` lines about
> "added jitter" are harmless BoTorch numerical notes.

### Train the models (optional, needs `.[ml]`)
```cmd
python -m sonoforge.plm.train --synthetic --epochs 5
python -m sonoforge.generative.train --steps 50 --n-res 8
```

---

## 8. Reproduce the benchmark
```cmd
python benchmarks\benchmark_optimizers.py --n-seeds 5 --n-cycles 6
```
Prints the mean-hypervolume table and writes `benchmarks\results.csv`.
Expected ordering: **qNEHVI > NSGA-II > random**.

---

## 9. Run the tests / lint (optional)
```cmd
pytest -q
ruff check src tests scripts benchmarks
```

## 10. Update to the latest version later
```cmd
cd sonoforge
git pull
pip install -e ".[dev]"
```

---

## The optimizers at a glance
| `--optimizer` | needs `.[ml]`? | what it is |
|---|:--:|---|
| `random`   | no  | random-search baseline |
| `nsga2`    | no  | genetic multi-objective (fast, no GPU) |
| `qnehvi`   | yes | constrained Bayesian optimization (best designs) |
| `gflownet` | yes | diversity-seeking reinforcement-learning generator |

---

## Other shells

**PowerShell** — same as cmd, but activate with:
```powershell
.\.venv\Scripts\Activate.ps1
#   if blocked: Set-ExecutionPolicy -Scope Process -Bypass   (then re-run)
```

**macOS / Linux:**
```bash
git clone https://github.com/sanjaydoc/sonoforge.git
cd sonoforge
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_cycle.py --optimizer nsga2 --n-cycles 6
```

---

## Troubleshooting

- **`activate.bat` opens Notepad / does nothing** → you're in cmd; use
  `.venv\Scripts\activate.bat`. (`Activate.ps1` is PowerShell-only.)
- **Activation won't run at all** → skip it and call the venv Python directly:
  ```cmd
  .venv\Scripts\python.exe -m pip install -e ".[dev]"
  .venv\Scripts\python.exe scripts\run_cycle.py --optimizer nsga2 --n-cycles 6
  ```
- **A dependency tries to *compile* and fails (common on brand-new Python)** →
  use Python **3.12** for the venv:
  ```cmd
  rmdir /s /q .venv
  py -3.12 -m venv .venv
  .venv\Scripts\activate.bat
  pip install -e ".[dev]"
  ```
- **Windows Defender popup during install** → it's reacting to a compiler; with
  the core install (all prebuilt wheels) it shouldn't appear. Allow it if asked.

Full annotated command reference: see **`How-To-Run-Commands.txt`**.
