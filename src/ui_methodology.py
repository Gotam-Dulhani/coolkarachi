"""Methodology and data dictionary, distilled from the paper."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import ui_common as c


def _targets() -> None:
    st.markdown(
        "The same pipeline is fitted to three target definitions. Reporting all three "
        "separates what a model learns about a neighbourhood from what is simply "
        "geography and weather."
    )
    cols = st.columns(3)
    with cols[0]:
        st.markdown("#### Local anomaly")
        st.latex(r"LST_{local} = LST - \overline{LST}_{block,\,year}")
        st.caption(
            "Removes inter-annual weather and the broad spatial gradient. The primary "
            "target for intervention design."
        )
    with cols[1]:
        st.markdown("#### Within-year anomaly")
        st.latex(r"LST_{anom} = LST - \overline{LST}_{year}")
        st.caption(
            "Keeps the city-wide spatial pattern, including the coastal–inland gradient."
        )
    with cols[2]:
        st.markdown("#### Raw LST")
        st.latex(r"LST \;\; (^\circ C)")
        st.caption(
            "Absolute temperature. `YEAR` is added as a predictor so the model can "
            "separate meteorology from geography."
        )

    st.write("")
    c.note(
        "A model with a high raw-LST R² is not necessarily useful for policy: most of "
        "that skill is the static coastal–inland climate gradient. The local anomaly is "
        "harder but answers the question that matters — which pixels are hotter than "
        "their immediate surroundings?"
    )


def _features() -> None:
    st.markdown(
        "Nine continuous predictors are used for the anomaly targets; raw LST adds "
        "`YEAR`. Land-cover class is deliberately excluded because it discretises "
        "information already carried by the continuous fractions and would make "
        "counterfactual perturbation incoherent."
    )
    rows = [
        ("LST", "°C", "Landsat 8/9 C2 L2 `ST_B10`", "Prediction target"),
        ("NDVI", "index", "Sentinel-2 (May–June median)", "Vegetation vigour"),
        ("NDBI", "index", "Sentinel-2", "Built-up intensity"),
        ("NDWI", "index", "Sentinel-2", "Surface moisture"),
        ("ALBEDO", "0–1", "Sentinel-2, Liang-style broadband", "Reflectivity"),
        ("BUILT_FRAC", "0–1", "Dynamic World", "Impervious fraction"),
        ("TREE_FRAC", "0–1", "Dynamic World", "Canopy fraction"),
        ("WATER_FRAC", "0–1", "Dynamic World", "Open-water fraction"),
        ("ELEV", "m", "SRTM 1 arc-second", "Terrain / coastal gradient"),
        ("POP_LOG", "log persons", "WorldPop 2020", "Exposure"),
        ("LON / LAT", "degrees", "Earth Engine pixel centroid", "Spatial blocking and maps"),
        ("YEAR", "—", "Observation year", "Used only for raw LST"),
    ]
    df = pd.DataFrame(rows, columns=["Column", "Unit", "Source", "Role"])
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.warning(
        "Known data issues: raw `ALBEDO` reaches 1.94 and is clipped to 0.02–0.65; "
        "`POP` is static across years because WorldPop provides one 2020 surface; "
        "`ELEV` and `POP` are collinear with distance from the coast, so elevation "
        "dominance reflects a climate gradient rather than a lapse-rate effect."
    )


def _validation() -> None:
    st.markdown("### Spatial validation protocol")
    cols = st.columns(4)
    steps = [
        ("2 km blocks", "The city is partitioned into 0.02° tiles — about 701 blocks in the built-up sample."),
        ("20% held out", "Blocks, not rows, are split: 303,387 training rows and 56,613 test rows."),
        ("Optuna inside CV", "Hyperparameters are tuned with 3-fold GroupKFold on training blocks only."),
        ("No leakage", "No test pixel is adjacent to a training pixel, so the error estimate is honest."),
    ]
    for col, (head, body) in zip(cols, steps):
        with col:
            st.markdown(f"**{head}**")
            st.caption(body)

    st.write("")
    c.note(
        "A random split would let a flexible model interpolate a pixel's immediate "
        "neighbours, inflating R². Blocking makes the reported error pessimistic but "
        "representative of performance in an unseen part of the city — the only "
        "performance that matters for extrapolation."
    )

    st.markdown("### Models compared")
    models = pd.DataFrame(
        [
            ("Random Forest", "Bagged decorrelated trees", "Robust, little preprocessing; memory-heavy at depth."),
            ("XGBoost", "Regularised gradient boosting", "Strong tabular baseline; native missing-value handling."),
            ("LightGBM", "Histogram-based leaf-wise boosting", "Fast at scale; used for SHAP attribution."),
            ("MLP", "Feed-forward neural network", "Non-tree reference; standardised inputs and early stopping."),
        ],
        columns=["Model", "Mechanism", "Why it is included"],
    )
    st.dataframe(models, hide_index=True, use_container_width=True)


def _explainability() -> None:
    st.markdown(
        "SHAP decomposes each prediction into additive feature contributions. Because "
        "the local-anomaly target is already in °C, a SHAP value reads directly as "
        "“this feature makes this pixel this many degrees hotter or cooler than its "
        "neighbourhood mean”."
    )
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Attribution**")
        st.caption(
            "Computed on LightGBM because a deep 600-tree forest is prohibitively slow "
            "to explain and the four families are tied within ~0.004 °C RMSE."
        )
        st.markdown("**Counterfactuals**")
        st.caption(
            "The fitted model is used as a surrogate: perturb NDVI, tree fraction and "
            "albedo on held-out pixels, re-predict, and report the change."
        )
    with cols[1]:
        st.markdown("**What is reported honestly**")
        st.caption(
            "The negative urban–rural SUHI, the weak and counter-intuitive albedo "
            "effect, the static population layer, and the difference between surface "
            "and air temperature."
        )
        st.markdown("**What is not claimed**")
        st.caption(
            "Causality. The scenarios are model-based counterfactuals, not field "
            "experiments, and surface temperature is not the same as the air "
            "temperature people experience."
        )


def _limitations() -> None:
    limitations = [
        ("Surface ≠ air temperature", "Landsat measures the radiating surface. Canopy-layer air temperature is buffered by advection and storage, so satellite LST cannot replace meteorological measurement."),
        ("Season window", "May–June only. The results describe pre-monsoon heat stress, not an annual average."),
        ("Static population", "WorldPop supplies one 2020 surface, reused for 2021–2025."),
        ("Albedo confounding", "Bright dry roofs and bare ground can be hot, so increasing albedo does not reliably cool in this climate."),
        ("Model-based counterfactuals", "Scenarios hold all other features fixed. Real interventions interact with moisture, shade, geometry and behaviour."),
        ("Spatial extrapolation", "Blocked validation is honest about unseen neighbourhoods but still assumes the fitted relationships transfer across the city."),
        ("No causal identification", "SHAP ranks associations; it does not prove that changing a feature will change temperature by the SHAP amount."),
    ]
    for head, body in limitations:
        st.markdown(f"**{head}.** {body}")


def render() -> None:
    c.hero(
        "Reproducible by design",
        "Method",
        "ology",
        "The target definitions, feature stack, spatial validation protocol and the "
        "explicit limitations behind the digital twin.",
    )
    tabs = st.tabs(["Targets", "Feature dictionary", "Validation & models", "Explainability", "Limitations"])
    with tabs[0]:
        _targets()
    with tabs[1]:
        _features()
    with tabs[2]:
        _validation()
    with tabs[3]:
        _explainability()
    with tabs[4]:
        _limitations()

    st.divider()
    c.section("Reproduce the pipeline")
    st.code(
        "python src/00_check.py\n"
        "python src/01_export_features.py --years 2020 2021 2022 2023 2024 2025\n"
        "python src/01_export_features.py --years 2020 2021 2022 2023 2024 2025 --urban-only --no-raster\n"
        "python src/02_eda.py --source urban\n"
        "python src/03_models.py --trials 15\n"
        "python src/03_models.py --target anomaly --trials 12\n"
        "python src/03_models.py --target raw --trials 12\n"
        "python src/04_paper_assets.py --inject paper/paper.md\n"
        "python src/05_build_paper.py\n"
        "python src/06_build_slides.py",
        language="powershell",
    )
