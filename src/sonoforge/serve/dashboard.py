"""Streamlit dashboard: run the loop and visualize HV trajectory + designs.

Requires ``streamlit`` (``pip install sonoforge[serve]``). Run with:
    streamlit run src/sonoforge/serve/dashboard.py
"""

from __future__ import annotations


def main() -> None:  # pragma: no cover - interactive
    import pandas as pd
    import streamlit as st

    from sonoforge.serve.service import SonoForgeService

    st.set_page_config(page_title="SonoForge", page_icon="🔊", layout="wide")
    st.title("🔊 SonoForge — closed-loop acoustic-reporter design")

    optimizer = st.sidebar.selectbox("Optimizer", ["nsga2", "qnehvi", "random"])
    n_cycles = st.sidebar.slider("DBTL cycles", 1, 15, 6)
    library_size = st.sidebar.slider("Library size", 4, 48, 16, step=4)

    if st.sidebar.button("Run design loop"):
        report = SonoForgeService().design(
            optimizer=optimizer, n_cycles=n_cycles, library_size=library_size
        )
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Feasible-front hypervolume")
            st.line_chart(pd.DataFrame({"hypervolume": report.hypervolume_trajectory}))
        with c2:
            st.subheader("Feasible fraction")
            st.line_chart(pd.DataFrame({"feasible_fraction": report.feasible_fraction}))
        st.subheader("Top feasible Pareto designs")
        st.dataframe(pd.DataFrame([{"sequence": d.sequence, **d.properties, "score": d.score}
                                   for d in report.designs]))


if __name__ == "__main__":
    main()
