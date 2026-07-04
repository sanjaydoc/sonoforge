"""Serving layer: typed service API + Gradio app + REST API + dashboard.

Only the torch/gradio-free :class:`SonoForgeService` is exported here; the app,
api, and dashboard import their heavy deps lazily.
"""

from sonoforge.serve.service import DesignReport, DesignResult, SonoForgeService

__all__ = ["DesignReport", "DesignResult", "SonoForgeService"]
