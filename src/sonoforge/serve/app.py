"""Gradio 'design-an-ARG' app — the non-expert front door.

Launches a small web UI: pick an optimizer and budget, click, and get ranked,
feasibility-checked acoustic-reporter designs with their predicted properties.

Requires ``gradio`` (``pip install sonoforge[serve]``).
"""

from __future__ import annotations

from sonoforge.serve.service import SonoForgeService


def build_demo():
    import gradio as gr

    service = SonoForgeService()

    def run(optimizer, n_cycles, library_size, top_k):
        report = service.design(
            optimizer=optimizer, n_cycles=int(n_cycles),
            library_size=int(library_size), top_k=int(top_k),
        )
        rows = [
            [d.sequence[:40] + ("…" if len(d.sequence) > 40 else ""),
             round(d.properties["contrast"], 3),
             round(d.properties["collapse_pressure"], 3),
             round(d.properties["immunogenicity"], 3),
             round(d.score, 3)]
            for d in report.designs
        ]
        hv = report.hypervolume_trajectory
        summary = (f"Evaluated {report.n_evaluated} candidates · "
                   f"HV {hv[0]:.3f} → {hv[-1]:.3f} · {len(report.designs)} designs returned")
        return summary, rows

    with gr.Blocks(title="SonoForge — design an acoustic reporter") as demo:
        gr.Markdown("# 🔊 SonoForge\nDesign gas-vesicle acoustic reporter genes with a closed-loop optimizer.")
        with gr.Row():
            optimizer = gr.Dropdown(["nsga2", "qnehvi", "random"], value="nsga2", label="Optimizer")
            n_cycles = gr.Slider(1, 15, value=6, step=1, label="DBTL cycles")
            library_size = gr.Slider(4, 48, value=16, step=4, label="Library size")
            top_k = gr.Slider(1, 20, value=10, step=1, label="Designs to return")
        btn = gr.Button("Design", variant="primary")
        summary = gr.Textbox(label="Run summary")
        table = gr.Dataframe(
            headers=["sequence", "contrast", "collapse_p", "immunogenicity", "score"],
            label="Top feasible Pareto designs",
        )
        btn.click(run, [optimizer, n_cycles, library_size, top_k], [summary, table])
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
