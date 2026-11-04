"""Model results: comparison, SHAP attribution and tuned hyperparameters."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import ui_common as c


def _metric_chart(df: pd.DataFrame, metric: str, colour: str, title: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("model:N", title=None, sort="-y", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{metric}:Q", title=title),
            color=alt.value(colour),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip(f"{metric}:Q", title=metric, format=".3f"),
                alt.Tooltip("n:Q", title="Test rows", format=","),
            ],
        )
        .properties(height=260)
    )


def _comparison(target: str) -> None:
    path = c.RESULTS / f"model_comparison_{target}.csv"
    df = c.load_csv(path)
    if df.empty:
        st.info(f"No comparison table for the **{c.TARGET_SHORT[target]}** target yet.")
        return

    df = df.sort_values("RMSE").reset_index(drop=True)
    best = df.iloc[0]

    kpis = st.columns(4)
    kpis[0].metric("Best model", str(best["model"]))
    kpis[1].metric("RMSE", f"{best['RMSE']:.3f} °C", help=c.METRIC_HELP["RMSE"])
    kpis[2].metric("R²", f"{best['R2']:.3f}", help=c.METRIC_HELP["R2"])
    kpis[3].metric("Mean bias", f"{best['MBE']:+.3f} °C", help=c.METRIC_HELP["MBE"])

    st.caption(
        f"Held-out evaluation on {c.fmt_int(best['n'])} pixels from spatially disjoint "
        "2 km blocks. Lower RMSE is better; R² is computed against the held-out blocks."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(
            _metric_chart(df, "RMSE", "#f97316", "RMSE (°C)"),
            use_container_width=True,
        )
    with col2:
        st.altair_chart(
            _metric_chart(df, "R2", "#0f766e", "R²"),
            use_container_width=True,
        )

    with st.expander("Full comparison table", expanded=True):
        show = df[["model", "RMSE", "MAE", "R2", "MBE", "n"]].copy()
        st.dataframe(
            show,
            hide_index=True,
            use_container_width=True,
            column_config={
                "model": "Model",
                "RMSE": st.column_config.NumberColumn("RMSE (°C)", format="%.3f"),
                "MAE": st.column_config.NumberColumn("MAE (°C)", format="%.3f"),
                "R2": st.column_config.NumberColumn("R²", format="%.3f"),
                "MBE": st.column_config.NumberColumn("MBE (°C)", format="%+.3f"),
                "n": st.column_config.NumberColumn("Test rows", format="%d"),
            },
        )
        st.caption(
            "Four model families — Random Forest, XGBoost, LightGBM and a multilayer "
            "perceptron — are statistically indistinguishable on the local target."
        )

    st.markdown("### Predicted vs observed")
    image = c.FIGS / f"pred_vs_obs_{target}.png"
    if image.exists():
        st.image(str(image), use_container_width=True)
        st.caption(
            "Hexbin density on held-out blocks; the red line is y = x. Tight clustering "
            "around the line indicates unbiased predictions."
        )
    else:
        st.info("Predicted-vs-observed figure not found.")

    st.markdown("### SHAP attribution")
    shap = c.load_csv(c.RESULTS / f"shap_importance_{target}.csv")
    if shap.empty:
        st.info("SHAP importance was not computed for this target.")
    else:
        shap = shap.sort_values("mean_abs_shap", ascending=False)
        chart = (
            alt.Chart(shap)
            .mark_bar(cornerRadiusEnd=4, color="#f97316")
            .encode(
                y=alt.Y("feature:N", title=None, sort="-x"),
                x=alt.X("mean_abs_shap:Q", title="Mean |SHAP| (°C)"),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("mean_abs_shap:Q", title="Mean |SHAP|", format=".4f"),
                ],
            )
            .properties(height=max(280, 30 * len(shap)))
        )
        st.altair_chart(chart, use_container_width=True)
        if target == "local":
            st.caption(
                "Because the local-anomaly target is in °C, SHAP values read directly as "
                "“this feature makes this pixel this many degrees hotter or cooler than its "
                "2 km neighbourhood”."
            )
        image = c.FIGS / f"shap_summary_{target}.png"
        if image.exists():
            with st.expander("SHAP beeswarm"):
                st.image(str(image), use_container_width=True)

    params = c.pretty_params(target)
    if not params.empty:
        with st.expander("Tuned hyperparameters (Optuna, GroupKFold)"):
            st.dataframe(params, hide_index=True, use_container_width=True)


def _all_targets() -> None:
    frames = []
    for target in c.available_targets():
        df = c.load_csv(c.RESULTS / f"model_comparison_{target}.csv")
        if df.empty:
            continue
        df = df.copy()
        df["target"] = c.TARGET_SHORT[target]
        frames.append(df)
    if not frames:
        st.info("No comparison tables found.")
        return

    combined = pd.concat(frames, ignore_index=True)
    chart = (
        alt.Chart(combined)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("RMSE:Q", title="RMSE (°C)"),
            color=alt.Color("target:N", title="Target"),
            xOffset="target:N",
            tooltip=[
                alt.Tooltip("target:N", title="Target"),
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("RMSE:Q", title="RMSE", format=".3f"),
                alt.Tooltip("R2:Q", title="R²", format=".3f"),
            ],
        )
        .properties(height=340, title="Error across the three prediction targets")
    )
    st.altair_chart(chart, use_container_width=True)

    st.dataframe(
        combined[["target", "model", "RMSE", "MAE", "R2", "MBE", "n"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "target": "Target",
            "model": "Model",
            "RMSE": st.column_config.NumberColumn("RMSE (°C)", format="%.3f"),
            "MAE": st.column_config.NumberColumn("MAE (°C)", format="%.3f"),
            "R2": st.column_config.NumberColumn("R²", format="%.3f"),
            "MBE": st.column_config.NumberColumn("MBE (°C)", format="%+.3f"),
            "n": st.column_config.NumberColumn("Test rows", format="%d"),
        },
    )
    c.note(
        "The local anomaly is the hardest target (lowest R²) but the most useful one: "
        "it removes weather and the coastal–inland gradient, leaving only the "
        "neighbourhood-scale signal that an intervention can change."
    )


def render() -> None:
    c.hero(
        "Spatially blocked evaluation",
        "Model",
        "Results",
        "Compare Random Forest, XGBoost, LightGBM and MLP across three LST targets, "
        "then read the SHAP attribution behind the best-performing family.",
    )

    targets = c.available_targets()
    if not targets:
        st.warning("No `model_comparison_*.csv` files found in `results/`.")
        return

    labels = [c.TARGET_SHORT[t] for t in targets]
    tab_objs = st.tabs(labels + ["All targets"])
    for tab, target in zip(tab_objs[:-1], targets):
        with tab:
            _comparison(target)
    with tab_objs[-1]:
        _all_targets()

    st.divider()
    c.section("Why the local target wins on usefulness, not on R²")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Local anomaly**")
        st.caption(
            "LST minus the same 2 km block-year mean. Removes weather and geography; "
            "answers “which pixels are hotter than their neighbours?”."
        )
    with cols[1]:
        st.markdown("**Within-year anomaly**")
        st.caption(
            "LST minus the year mean. Keeps the city-wide spatial pattern but removes "
            "inter-annual weather."
        )
    with cols[2]:
        st.markdown("**Raw LST**")
        st.caption(
            "Absolute temperature. Adds `YEAR` as a predictor; high R² is dominated by "
            "the coastal–inland climate gradient."
        )
