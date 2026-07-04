"""Phase 6 tests: the design service, benchmark, and (optional) API/app builders."""

import pytest

from sonoforge.benchmark import BenchmarkResult, format_table, run_benchmark
from sonoforge.serve import DesignReport, SonoForgeService

# --- service ---------------------------------------------------------------

def test_service_design_returns_ranked_feasible_designs():
    report = SonoForgeService().design(n_cycles=3, library_size=12, n_seed=12, top_k=5, seed=0)
    assert isinstance(report, DesignReport)
    assert len(report.hypervolume_trajectory) == 4
    assert report.n_evaluated > 0
    assert len(report.designs) <= 5
    if report.designs:
        # returned designs are feasible and sorted by score (desc)
        assert all(d.feasible for d in report.designs)
        scores = [d.score for d in report.designs]
        assert scores == sorted(scores, reverse=True)
        d = report.designs[0]
        assert set(d.properties) == {
            "contrast", "collapse_pressure", "expressibility", "solubility", "immunogenicity"
        }


def test_service_report_is_json_serializable():
    import json

    report = SonoForgeService().design(n_cycles=2, library_size=8, n_seed=8)
    json.dumps(report.to_dict())  # must not raise


# --- benchmark -------------------------------------------------------------

def test_run_benchmark_random_vs_nsga2():
    results = run_benchmark(optimizers=["random", "nsga2"], n_seeds=2, n_cycles=3,
                            library_size=12, n_seed_lib=12)
    assert len(results) == 2
    assert all(isinstance(r, BenchmarkResult) for r in results)
    assert all(r.final_hv_mean >= 0 for r in results)
    table = format_table(results)
    assert "optimizer" in table and "nsga2" in table


# --- optional serve backends ----------------------------------------------

def test_fastapi_app_builds_if_installed():
    pytest.importorskip("fastapi")
    from sonoforge.serve.api import create_app

    app = create_app()
    routes = {r.path for r in app.routes}
    assert "/design" in routes and "/health" in routes


def test_gradio_demo_builds_if_installed():
    pytest.importorskip("gradio")
    from sonoforge.serve.app import build_demo

    assert build_demo() is not None
