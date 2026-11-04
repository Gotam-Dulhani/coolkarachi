"""Cooling scenarios: published counterfactuals + optional live simulator."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import ui_common as c


def _precomputed() -> None:
    st.markdown("### Published counterfactual results")
    st.caption(
        "Each scenario is applied to the held-out test pixels and re-predicted with the "
        "fitted surrogate. Negative ΔLST means cooling."
    )

    targets = c.available_targets()
    if not targets:
        st.warning("No scenario result files found in `results/`.")
        return

    target = st.selectbox(
        "Prediction target",
        targets,
        format_func=lambda t: c.TARGET_LABELS[t],
        key="scenario_target",
    )
    df = c.load_csv(c.RESULTS / f"scenarios_{target}.csv")
    if df.empty:
        st.info("No scenario table for this target.")
        return

    df = df.copy()
    df["label"] = df["scenario"].map(c.SCENARIO_LABELS).fillna(df["scenario"].str.replace("_", " "))
    best = df.sort_values("mean_delta_C").iloc[0]

    kpis = st.columns(4)
    kpis[0].metric("Strongest mean cooling", f"{best['mean_delta_C']:+.2f} °C")
    kpis[1].metric("Median ΔLST", f"{best['median_delta_C']:+.2f} °C")
    kpis[2].metric("Pixels < −0.5 °C", f"{best['pct_cooling_gt_0.5C']:.1f}%")
    kpis[3].metric("Pixels < −1.0 °C", f"{best['pct_cooling_gt_1C']:.1f}%")
    st.caption(f"Best scenario: **{best['label']}**")

    col1, col2 = st.columns(2)
    with col1:
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                y=alt.Y("label:N", title=None, sort="x"),
                x=alt.X("mean_delta_C:Q", title="Mean ΔLST (°C)"),
                color=alt.condition(
                    alt.datum.mean_delta_C < 0, alt.value("#0f766e"), alt.value("#b91c1c")
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="Scenario"),
                    alt.Tooltip("mean_delta_C:Q", title="Mean ΔLST", format="+.2f"),
                    alt.Tooltip("median_delta_C:Q", title="Median ΔLST", format="+.2f"),
                ],
            )
            .properties(height=300, title="Mean temperature change")
        )
        st.altair_chart(chart, use_container_width=True)

    with col2:
        long = df.melt(
            id_vars=["label"],
            value_vars=["pct_cooling_gt_0.5C", "pct_cooling_gt_1C"],
            var_name="threshold",
            value_name="percent",
        )
        long["threshold"] = long["threshold"].map(
            {"pct_cooling_gt_0.5C": "Cooling > 0.5 °C", "pct_cooling_gt_1C": "Cooling > 1.0 °C"}
        )
        chart = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                x=alt.X("threshold:N", title=None),
                y=alt.Y("percent:Q", title="% of test pixels"),
                color=alt.Color("threshold:N", legend=None),
                xOffset="label:N",
                tooltip=[
                    alt.Tooltip("label:N", title="Scenario"),
                    alt.Tooltip("threshold:N", title="Threshold"),
                    alt.Tooltip("percent:Q", title="% pixels", format=".1f"),
                ],
            )
            .properties(height=300, title="Share of pixels crossing a cooling threshold")
        )
        st.altair_chart(chart, use_container_width=True)

    st.dataframe(
        df[["label", "mean_delta_C", "median_delta_C", "pct_cooling_gt_0.5C", "pct_cooling_gt_1C"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "label": "Scenario",
            "mean_delta_C": st.column_config.NumberColumn("Mean ΔLST (°C)", format="%+.2f"),
            "median_delta_C": st.column_config.NumberColumn("Median ΔLST (°C)", format="%+.2f"),
            "pct_cooling_gt_0.5C": st.column_config.NumberColumn("Cells < −0.5 °C (%)", format="%.1f"),
            "pct_cooling_gt_1C": st.column_config.NumberColumn("Cells < −1.0 °C (%)", format="%.1f"),
        },
    )

    if target == "local":
        c.note(
            "On the local target, greening cools the average pixel by about 0.6 °C and "
            "the combined scenario barely improves on greening alone. Albedo is nearly "
            "inert because bright dry surfaces in Karachi are also hot — the standard "
            "cool-roof assumption does not transfer cleanly to this climate."
        )


def _get_prepared(df: pd.DataFrame) -> pd.DataFrame:
    """The explorer stores an already-prepared frame; only redo it if needed."""
    if {"POP_LOG", "LST_ANOM"}.issubset(df.columns):
        return df
    return c.prepare_dataset(df)


def _model_for_target(target: str, df: pd.DataFrame):
    """Return (model, features, description) or (None, features, '') ."""
    prepared = _get_prepared(df)
    features = c.feature_list(target, prepared)
    ok, message = c.dataset_ready(target, prepared)
    if not ok:
        st.warning(message)
        return None, features, prepared

    state_key = f"ck_model_{target}"
    if state_key in st.session_state:
        model, saved_features = st.session_state[state_key]
        return model, saved_features, prepared

    caches = c.model_caches(target)
    if caches:
        choice = st.selectbox(
            "Model artefact",
            [p.name for p in caches],
            key=f"model_choice_{target}",
            help="Cached models from `03_models.py`, or a surrogate trained in this session.",
        )
        path = next(p for p in caches if p.name == choice)
        try:
            model = c.load_model(path)
            return model, features, prepared
        except Exception as exc:
            st.error(f"Could not load `{path.name}`: {exc}")
            return None, features, prepared

    st.info(
        "No fitted model is cached locally. Train a lightweight surrogate from the loaded "
        "sample to explore the sliders. It is for orientation only, not a replacement for "
        "the paper's spatially validated models."
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        rows = st.slider(
            "Training rows",
            min_value=5_000,
            max_value=200_000,
            value=80_000,
            step=5_000,
            disabled=len(prepared) < 5_000,
        )
    with col2:
        st.write("")
        st.write("")
        train = st.button("Train quick surrogate", type="primary", use_container_width=True)
    if train:
        with st.spinner("Fitting LightGBM on the loaded sample…"):
            try:
                model, saved_features, n = c.train_quick_surrogate(df, target, rows)
                c.save_surrogate(model, target)
                st.session_state[state_key] = (model, saved_features)
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                return None, features, prepared
        st.success(f"Surrogate trained on {n:,} rows and saved to `results/ui_surrogate_{target}.joblib`.")
        return model, saved_features, prepared

    return None, features, prepared


def _simulator() -> None:
    st.markdown("### Live what-if simulator")
    df = c.get_dataset()
    if df is None:
        c.warning(
            "Load a CSV export in **Data explorer** first. The simulator needs baseline "
            "pixels to perturb; the published results above do not require data."
        )
        st.caption("Expected path: `data/karachi_ml_<year>.csv` with LON/LAT and YEAR columns.")
        return

    targets = c.available_targets()
    if not targets:
        st.warning("No target definitions available.")
        return

    target = st.selectbox(
        "Simulation target",
        targets,
        format_func=lambda t: c.TARGET_LABELS[t],
        key="sim_target",
    )
    model, features, prepared = _model_for_target(target, df)
    if model is None:
        return

    required = features + [c.TARGETS[target]]
    base = prepared.dropna(subset=required).copy()
    if base.empty:
        st.warning("No complete baseline rows for this target.")
        return

    restrict = True
    if "BUILT_FRAC" in base.columns:
        restrict = st.checkbox(
            "Restrict baseline to built-up pixels (BUILT_FRAC ≥ 0.2)",
            value=True,
            help="The paper's ML sample is already built-up; this also protects against "
            "mixing rural pixels when a full-cover CSV is loaded.",
        )
    if restrict and "BUILT_FRAC" in base.columns:
        base = base[base["BUILT_FRAC"] >= 0.2]
    if base.empty:
        st.warning("No baseline rows remain after the built-up filter.")
        return

    baseline = c.sample_frame(base, 5_000)
    st.caption(
        f"Perturbing {len(baseline):,} baseline pixels · model features: "
        + ", ".join(features)
    )

    st.markdown("**Intervention**")
    col1, col2, col3 = st.columns(3)
    ndvi = col1.slider("NDVI increase", 0.00, 0.30, 0.10, 0.01)
    tree = col2.slider("Tree-fraction increase", 0.00, 0.20, 0.05, 0.01)
    albedo = col3.slider("Albedo increase", 0.00, 0.30, 0.15, 0.01)

    deltas = {"NDVI": ndvi, "TREE_FRAC": tree, "ALBEDO": albedo}
    _, delta = c.predict_delta(model, baseline, features, deltas)
    metrics = c.scenario_metrics(delta)

    st.write("")
    kpis = st.columns(4)
    kpis[0].metric("Mean ΔLST", f"{metrics['mean_delta']:+.2f} °C")
    kpis[1].metric("Median ΔLST", f"{metrics['median_delta']:+.2f} °C")
    kpis[2].metric("Pixels < −0.5 °C", f"{metrics['pct_cool_05']:.1f}%")
    kpis[3].metric("Pixels < −1.0 °C", f"{metrics['pct_cool_10']:.1f}%")

    if metrics["mean_delta"] < 0:
        c.note(
            f"This intervention cools the average pixel by "
            f"<b>{abs(metrics['mean_delta']):.2f} °C</b>, and "
            f"<b>{metrics['pct_cool_10']:.1f}%</b> of pixels cool by more than 1 °C."
        )
    else:
        c.warning(
            "This combination warms the average pixel. In this climate, brightening dry "
            "surfaces can raise surface temperature; check the SHAP attribution."
        )

    if {"LON", "LAT"}.issubset(baseline.columns):
        map_df = pd.DataFrame(
            {
                "LON": pd.to_numeric(baseline["LON"], errors="coerce").to_numpy(),
                "LAT": pd.to_numeric(baseline["LAT"], errors="coerce").to_numpy(),
                "delta": delta,
            }
        ).dropna()
        if "LST" in baseline.columns:
            map_df["LST"] = pd.to_numeric(baseline["LST"], errors="coerce").to_numpy()
        limit = float(np.nanquantile(np.abs(map_df["delta"]), 0.99)) if len(map_df) else 1.0
        limit = max(limit, 0.5)
        chart = (
            alt.Chart(map_df)
            .mark_circle(size=14, opacity=0.65)
            .encode(
                x=alt.X("LON:Q", title="Longitude", scale=alt.Scale(zero=False)),
                y=alt.Y("LAT:Q", title="Latitude", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "delta:Q",
                    scale=alt.Scale(
                        scheme="redblue",
                        domain=[-limit, limit],
                        reverse=True,
                    ),
                    title="ΔLST (°C)",
                ),
                tooltip=[
                    alt.Tooltip("LON:Q", title="Lon", format=".5f"),
                    alt.Tooltip("LAT:Q", title="Lat", format=".5f"),
                    alt.Tooltip("delta:Q", title="ΔLST", format="+.2f"),
                ],
            )
            .properties(height=520, title="Predicted change on the loaded sample")
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption("Blue = cooling, red = warming. Point cloud in geographic coordinates.")

    export = baseline[["LON", "LAT", "LST"]].copy() if {"LON", "LAT", "LST"}.issubset(baseline.columns) else pd.DataFrame()
    if not export.empty:
        export["delta_LST_C"] = delta
    else:
        export = pd.DataFrame({"delta_LST_C": delta})
    st.download_button(
        "Download per-pixel scenario",
        export.to_csv(index=False).encode("utf-8"),
        file_name=f"coolkarachi_scenario_{target}.csv",
        mime="text/csv",
    )

    with st.expander("How the simulator works"):
        st.markdown(
            "- Baseline pixels are sampled from the loaded export (up to 5,000 rows).\n"
            "- The model predicts each pixel before and after the three perturbations, "
            "clipping NDVI, tree fraction and albedo to their physical limits.\n"
            "- ΔLST is the difference between those predictions, so negative values mean "
            "cooling.\n"
            "- For the **raw** target the model also receives `YEAR`; the perturbation does "
            "not change the year.\n"
            "- A surrogate trained here uses a random 80/20 holdout internally only for "
            "speed; treat its absolute values as indicative."
        )


def render() -> None:
    c.hero(
        "Digital-twin counterfactuals",
        "Cooling",
        "Scenarios",
        "Read the validated greening and cool-roof experiments, then perturb a loaded "
        "sample yourself and watch the predicted temperature change.",
    )
    _precomputed()
    st.divider()
    _simulator()
