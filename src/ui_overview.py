"""Landing page: project story, headline numbers and artefact health."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import ui_common as c

ORANGE = "#f97316"
TEAL = "#0f766e"
RED = "#b91c1c"


def _best(target: str):
    return c.best_model(target)


def render() -> None:
    c.hero(
        "Karachi · pre-monsoon 2020–2025 · Landsat 8/9 at 30 m",
        "CoolKarachi",
        "Digital Twin",
        "An explainable machine-learning surrogate that maps surface heat, ranks its "
        "drivers, and simulates where greening and cool roofs buy the most cooling.",
    )

    c.status_pills()
    st.write("")

    suhi = c.load_csv(c.RESULTS / "suhi_full_sample.csv")
    has_suhi = not suhi.empty and "period" in suhi.columns
    years_df = suhi[suhi["period"] != "pooled"] if has_suhi else pd.DataFrame()
    pooled = suhi[suhi["period"] == "pooled"].iloc[0] if has_suhi and (suhi["period"] == "pooled").any() else None
    best_local = _best("local")
    best_raw = _best("raw")
    scenarios = c.load_csv(c.RESULTS / "scenarios_local.csv")

    # ------------------------------------------------------------------ KPIs
    cols = st.columns(4)
    if pooled is not None:
        cols[0].metric(
            "SUHI (°C)",
            c.fmt_metric(pooled.get("suhi_C"), 2, signed=True),
            help="Urban minus rural LST. Negative means the rural desert reference is hotter than the built-up surface.",
        )
        cols[1].metric(
            "Urban LST (°C)",
            c.fmt_metric(pooled.get("lst_urban_C"), 1),
            help="Built-up pixels, May–June 2020–2025.",
        )
    else:
        cols[0].metric("SUHI (°C)", "—")
        cols[1].metric("Urban LST (°C)", "—")

    if best_local is not None:
        cols[2].metric(
            "Best RMSE (°C)",
            c.fmt_metric(best_local["RMSE"], 3),
            help=f"{best_local['model']} on held-out 2 km spatial blocks.",
        )
    else:
        cols[2].metric("Best RMSE (°C)", "—")

    cols[3].metric(
        "Years",
        f"{len(years_df)}" if not years_df.empty else "6",
        help="2020–2025, May–June only.",
    )

    st.write("")

    # ------------------------------------------------------------- the story
    left, right = st.columns([1.45, 1])

    with left:
        c.section("The result that frames everything")
        if pooled is not None and years_df is not None and not years_df.empty:
            chart_df = years_df.copy()
            chart_df["suhi_C"] = pd.to_numeric(chart_df["suhi_C"], errors="coerce")
            bars = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("period:N", title="Year", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("suhi_C:Q", title="SUHI (°C)"),
                    color=alt.condition(
                        alt.datum.suhi_C < 0,
                        alt.value(TEAL),
                        alt.value(RED),
                    ),
                    tooltip=[
                        alt.Tooltip("period:N", title="Year"),
                        alt.Tooltip("suhi_C:Q", title="SUHI (°C)", format="+.2f"),
                        alt.Tooltip("lst_urban_C:Q", title="Urban LST", format=".2f"),
                        alt.Tooltip("lst_rural_C:Q", title="Rural LST", format=".2f"),
                    ],
                )
                .properties(height=285, title="Surface urban heat island by year")
            )
            rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#666").encode(y="y:Q")
            st.altair_chart(bars + rule, use_container_width=True)
            c.note(
                f"Karachi's surface heat island is <b>negative</b>: "
                f"{c.fmt_metric(pooled['suhi_C'], 2, signed=True)} °C pooled. "
                "The inland desert reference is hotter than the city, so urban–rural "
                "contrast is a poor intervention target. The project instead models "
                "<b>block-relative</b> heat."
            )
        else:
            st.info("SUHI summary not found in results/.")

    with right:
        c.section("Headline interventions")
        if not scenarios.empty:
            sc = scenarios.copy()
            sc["label"] = sc["scenario"].map(c.SCENARIO_LABELS).fillna(
                sc["scenario"].str.replace("_", " ")
            )
            short = {
                "greening_+10%_NDVI": "Greening",
                "cool_roof_+0.15_albedo": "Cool roof",
                "combined": "Combined",
            }
            sc["short"] = sc["scenario"].map(short).fillna(sc["label"])
            sc = sc.sort_values("mean_delta_C")
            chart = (
                alt.Chart(sc)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    y=alt.Y("short:N", title=None, sort="x"),
                    x=alt.X("mean_delta_C:Q", title="Mean ΔLST (°C)"),
                    color=alt.condition(
                        alt.datum.mean_delta_C < 0, alt.value(TEAL), alt.value(RED)
                    ),
                    tooltip=[
                        alt.Tooltip("label:N", title="Scenario"),
                        alt.Tooltip("mean_delta_C:Q", title="Mean ΔLST", format="+.2f"),
                        alt.Tooltip("pct_cooling_gt_1C:Q", title="% < −1 °C", format=".1f"),
                    ],
                )
                .properties(height=285, title="Scenario cooling")
            )
            st.altair_chart(chart, use_container_width=True)
            if len(sc):
                best = sc.iloc[0]
                st.caption(
                    f"Largest mean cooling: **{best['label']}** "
                    f"({c.fmt_metric(best['mean_delta_C'], 2, signed=True)} °C); "
                    f"{c.fmt_metric(best['pct_cooling_gt_1C'], 1)}% of pixels cool by more than 1 °C."
                )
        else:
            st.info("Scenario results not found in results/.")

    st.divider()

    # ----------------------------------------------------------- how it works
    c.section("How the twin is built")
    steps = st.columns(5)
    step_text = [
        ("01", "Earth Engine stack", "Landsat LST, Sentinel-2 indices, Dynamic World fractions, SRTM, WorldPop."),
        ("02", "Two samples", "360k built-up points for ML; full-cover sample for the urban–rural contrast."),
        ("03", "Spatial validation", "2 km blocks, 20% held out, Optuna tuned inside GroupKFold."),
        ("04", "Explain + perturb", "SHAP attribution and counterfactual greening / cool-roof scenarios."),
        ("05", "Decision support", "Neighbourhood-relative heat maps and cooling priorities."),
    ]
    for col, (num, head, body) in zip(steps, step_text):
        with col:
            st.markdown(f"**{num} · {head}**")
            st.caption(body)

    st.divider()

    # ------------------------------------------------------------- key findings
    left, right = st.columns([1.1, 1])
    with left:
        c.section("What the models say")
        shap = c.load_csv(c.RESULTS / "shap_importance_local.csv")
        bullets = []
        if not shap.empty:
            top = shap.iloc[0]
            bullets.append(
                f"**{top['feature'].replace('_', ' ').title()}** is the strongest local driver "
                f"(mean |SHAP| = {c.fmt_metric(top['mean_abs_shap'], 3)} °C)."
            )
        if best_local is not None:
            bullets.append(
                f"All four model families are effectively tied on the local target "
                f"(best RMSE {c.fmt_metric(best_local['RMSE'], 3)} °C, "
                f"R² {c.fmt_metric(best_local['R2'], 3)})."
            )
        if best_raw is not None:
            bullets.append(
                f"Raw LST is easier to predict (R² {c.fmt_metric(best_raw['R2'], 3)}), but most of "
                "that skill is the coastal–inland climate gradient, not a lever for intervention."
            )
        bullets.append(
            "Albedo is **not** a reliable cooling lever in this climate: bright dry surfaces "
            "are also hot, which contradicts the standard cool-roof narrative."
        )
        st.markdown("\n".join(f"- {b}" for b in bullets))

    with right:
        c.section("Artefact health")
        status = c.artifact_status()
        status["Found"] = status["Found"].apply(
            lambda v: f"✓ {v}" if v else "— missing"
        )
        st.dataframe(status, hide_index=True, use_container_width=True)
        st.caption(
            "CSV exports and fitted `.joblib` models are git-ignored. "
            "Load them in **Data explorer** to unlock the live what-if simulator."
        )

    with st.expander("Run the pipeline from a terminal", expanded=False):
        st.code(
            "python src/00_check.py\n"
            "python src/01_export_features.py --years 2020 2021 2022 2023 2024 2025\n"
            "python src/02_eda.py --source urban\n"
            "python src/03_models.py --trials 15\n"
            "python src/04_paper_assets.py --inject paper/paper.md\n"
            "python src/05_build_paper.py\n"
            "python src/06_build_slides.py",
            language="powershell",
        )
        st.caption(
            "Dashboard: `streamlit run app.py` (use the same virtual environment)."
        )
