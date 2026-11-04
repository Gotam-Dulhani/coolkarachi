"""Spatial views: per-year heat maps, scenario map and the figure library."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import ui_common as c


def _download_image(path) -> None:
    data = c.file_bytes(path)
    if data:
        st.download_button(
            "Download PNG",
            data,
            file_name=path.name,
            mime="image/png",
            key=f"dl_{path.name}",
        )


def _heatmaps() -> None:
    years = sorted(
        int(p.stem.split("_")[-1])
        for p in c.FIGS.glob("heatmap_*.png")
        if p.stem.split("_")[-1].isdigit()
    )
    if not years:
        st.info("No heat maps found in `figures/`.")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        year = st.selectbox("Year", years, index=len(years) - 1, key="heatmap_year")
    path = c.FIGS / f"heatmap_{year}.png"
    st.image(str(path), use_container_width=True)
    st.caption(
        "Left: absolute LST (°C). Right: local anomaly (LST minus the 2 km block-year "
        "mean). The anomaly panel is the decision-relevant view — it shows which pixels "
        "are hotter than their own neighbourhood."
    )
    _download_image(path)

    st.divider()
    st.markdown("### Year-to-year change")
    first, last = years[0], years[-1]
    col1, col2 = st.columns(2)
    for col, y in ((col1, first), (col2, last)):
        p = c.FIGS / f"heatmap_{y}.png"
        if p.exists():
            col.image(str(p), caption=f"{y}", use_container_width=True)


def _scenario_map() -> None:
    path = c.FIGS / "scenario_greening_map.png"
    if not path.exists():
        st.info("Scenario map not found. Run `python src/03_models.py` to create it.")
        return
    st.image(str(path), use_container_width=True)
    st.caption(
        "Predicted cooling from +0.10 NDVI greening on the held-out test sample. "
        "The map is a point cloud at 30 m pixel centroids, coloured by ΔLST."
    )
    _download_image(path)
    c.note(
        "Greening produces the largest and most reliable cooling signal. Cool-roof "
        "albedo is much weaker in Karachi because bright dry surfaces are also hot."
    )


def _suhi() -> None:
    df = c.load_csv(c.RESULTS / "suhi_full_sample.csv")
    if df.empty:
        st.info("SUHI summary not found.")
        return

    per_year = df[df["period"] != "pooled"].copy()
    per_year["suhi_C"] = pd.to_numeric(per_year["suhi_C"], errors="coerce")
    pooled = df[df["period"] == "pooled"]

    if not pooled.empty:
        row = pooled.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Urban mean LST", f"{row['lst_urban_C']:.2f} °C")
        cols[1].metric("Rural mean LST", f"{row['lst_rural_C']:.2f} °C")
        cols[2].metric("Pooled SUHI", f"{row['suhi_C']:+.2f} °C")
        cols[3].metric("Urban / rural pixels", f"{int(row['n_urban']):,} / {int(row['n_rural']):,}")

    bars = (
        alt.Chart(per_year)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("period:N", title="Year", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("suhi_C:Q", title="Urban − rural LST (°C)"),
            color=alt.condition(alt.datum.suhi_C < 0, alt.value("#0f766e"), alt.value("#b91c1c")),
            tooltip=[
                alt.Tooltip("period:N", title="Year"),
                alt.Tooltip("suhi_C:Q", title="SUHI (°C)", format="+.2f"),
                alt.Tooltip("lst_urban_C:Q", title="Urban LST", format=".2f"),
                alt.Tooltip("lst_rural_C:Q", title="Rural LST", format=".2f"),
            ],
        )
        .properties(height=360, title="Surface urban heat island by year")
    )
    rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#666").encode(y="y:Q")
    st.altair_chart(bars + rule, use_container_width=True)

    image = c.FIGS / "suhi_annual.png"
    if image.exists():
        st.image(str(image), use_container_width=True)
        _download_image(image)

    c.warning(
        "The urban–rural SUHI is negative every year (−1.1 to −1.9 °C). The inland "
        "desert reference is hotter than the built-up city, so this contrast cannot be "
        "used to rank neighbourhood interventions. The model therefore targets the "
        "block-relative local anomaly."
    )


def _library() -> None:
    figures = sorted(c.FIGS.glob("*.png"))
    if not figures:
        st.info("No figures found.")
        return
    names = [p.name for p in figures]
    choice = st.selectbox("Figure", names, key="figure_library")
    path = c.FIGS / choice
    st.image(str(path), use_container_width=True)
    _download_image(path)
    st.caption(f"`figures/{choice}`")


def render() -> None:
    c.hero(
        "30 m spatial evidence",
        "Heat",
        "Maps",
        "Per-year observed LST and neighbourhood anomalies, the greening scenario map, "
        "and the annual urban–rural contrast.",
    )
    tabs = st.tabs(["Yearly heat maps", "Greening scenario", "Urban–rural SUHI", "Figure library"])
    with tabs[0]:
        _heatmaps()
    with tabs[1]:
        _scenario_map()
    with tabs[2]:
        _suhi()
    with tabs[3]:
        _library()
