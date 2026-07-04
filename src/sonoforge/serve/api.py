"""FastAPI REST wrapper around the design service (model democratization).

Exposes ``POST /design`` returning ranked designs as JSON, so the model can be
served to non-domain experts and downstream systems.

Requires ``fastapi`` + ``uvicorn`` (``pip install sonoforge[serve]``).
"""

from __future__ import annotations

from sonoforge.serve.service import SonoForgeService


def create_app():
    from fastapi import FastAPI
    from pydantic import BaseModel

    class DesignRequest(BaseModel):
        seeds: list[str] | None = None
        optimizer: str = "nsga2"
        n_cycles: int = 5
        library_size: int = 16
        top_k: int = 10

    app = FastAPI(title="SonoForge API", version="0.1.0")
    service = SonoForgeService()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/design")
    def design(req: DesignRequest) -> dict:
        report = service.design(
            seeds=req.seeds, optimizer=req.optimizer, n_cycles=req.n_cycles,
            library_size=req.library_size, top_k=req.top_k,
        )
        return report.to_dict()

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
