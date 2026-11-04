"""Data explorer: load exports, filter, chart and map the underlying samples."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import ui_common as c

NUMERIC_HINTS = [
    "LST",
    "LST_ANOM",
    "LST_LOCAL",
    "NDVI",
    "NDBI",
    "NDWI",
    "ALBEDO",
    "BUILT_FRAC",
    "TREE_FRAC",
    "WATER_FRAC",
    "ELEV",
    "POP",
    "POP_LOG",
]


def _load_panel() -> None:
    """File picker / uploader. Stores the prepared frame in session state."""
    st.markdown("### 1 · Bring in a sample")
    st.caption(
        "The exports are not committed to the repository. Load `karachi_ml_*.csv` "
        "(built-up ML sample), `karachi_features_*.csv` (all-cover SUHI sample) or "
        "`karachi_urban_clean.csv`."
    )

    tab_disk, tab_upload = st.tabs(["Read from data/", "Upload CSV files"])

    with tab_disk:
        files = c.data_files()
        if not files:
            st.info(
                "No `karachi_*.csv` files in `data/`. Download the twelve yearly exports "
                "from the Google Drive folder, or upload them in the next tab."
            )
            st.link_button(
                "Open dataset folder",
                "https://drive.google.com/drive/folders/1ZbUw7r6Sor7C47uXzFwsOGu8laFZ32Cg",
            )
        else:
            options = [f.name for f in files]
            selected = st.multiselect(
                "CSV files",
                options,
                default=options[-1:],
                help="Start with one year; add more for a multi-year view.",
            )
            if st.button("Load selected files", type="primary", disabled=not selected):
                frames = []
                progress = st.progress(0.0, text="Reading CSVs…")
                for i, name in enumerate(selected, start=1):
                    frames.append(c.load_csv(c.DATA / name))
                    progress.progress(i / len(selected), text=f"Reading {name}")
                raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                with st.spinner("Cleaning and deriving model features…"):
                    prepared = c.prepare_dataset(raw)
                c.store_dataset(prepared, f"data/ → {', '.join(selected)}")
                progress.empty()
                st.rerun()

    with tab_upload:
        uploads = st.file_uploader(
            "Upload one or more CSV exports",
            type=["csv"],
            accept_multiple_files=True,
            help="Expected columns: LST, NDVI, NDBI, NDWI, ALBEDO, BUILT_FRAC, "
            "TREE_FRAC, WATER_FRAC, ELEV, POP, LON, LAT, YEAR, LULC.",
        )
        if st.button("Load uploaded files", type="primary", disabled=not uploads):
            with st.spinner("Reading and cleaning uploads…"):
                raw = c.read_uploaded(uploads)
                prepared = c.prepare_dataset(raw)
            c.store_dataset(prepared, f"upload → {', '.join(f.name for f in uploads)}")
            st.rerun()


def _histogram(df: pd.DataFrame, column: str) -> None:
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        st.info("No numeric values to plot.")
        return
    counts, edges = np.histogram(series, bins=55)
    hist = pd.DataFrame(
        {
            "value": (edges[:-1] + edges[1:]) / 2,
            "count": counts,
        }
    )
    chart = (
        alt.Chart(hist)
        .mark_bar(color="#f97316", opacity=0.85)
        .encode(
            x=alt.X("value:Q", title=column),
            y=alt.Y("count:Q", title="Pixels"),
            tooltip=[
                alt.Tooltip("value:Q", title=column, format=".2f"),
                alt.Tooltip("count:Q", title="Count", format=","),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"n = {len(series):,} · mean {series.mean():.2f} · median {series.median():.2f} "
        f"· SD {series.std():.2f} · min {series.min():.2f} · max {series.max():.2f}"
    )


def _correlation(df: pd.DataFrame) -> None:
    numeric = [col for col in NUMERIC_HINTS if col in df.columns]
    numeric = [col for col in numeric if pd.to_numeric(df[col], errors="coerce").notna().sum() > 10]
    if len(numeric) < 2:
        st.info("Not enough numeric columns for a correlation matrix.")
        return
    corr = df[numeric].apply(pd.to_numeric, errors="coerce").corr().round(3)
    long = corr.reset_index().melt(id_vars="index", var_name="feature", value_name="r")
    long.columns = ["row", "column", "r"]
    heat = (
        alt.Chart(long)
        .mark_rect()
        .encode(
            x=alt.X("column:N", title=None, sort=numeric),
            y=alt.Y("row:N", title=None, sort=numeric),
            color=alt.Color(
                "r:Q",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True),
                title="Pearson r",
            ),
            tooltip=[
                alt.Tooltip("row:N", title="Row"),
                alt.Tooltip("column:N", title="Column"),
                alt.Tooltip("r:Q", title="r", format=".3f"),
            ],
        )
        .properties(height=520)
    )
    st.altair_chart(heat, use_container_width=True)


def _scatter(df: pd.DataFrame, x: str, y: str) -> None:
    sample = c.sample_frame(df, 6000)
    chart = (
        alt.Chart(sample)
        .mark_circle(size=9, opacity=0.35)
        .encode(
            x=alt.X(f"{x}:Q", title=x),
            y=alt.Y(f"{y}:Q", title=y),
            color=alt.Color(f"{y}:Q", scale=alt.Scale(scheme="inferno"), legend=None),
            tooltip=[
                alt.Tooltip(f"{x}:Q", title=x, format=".3f"),
                alt.Tooltip(f"{y}:Q", title=y, format=".2f"),
            ],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"Random sample of {len(sample):,} rows.")


def _eda_gallery() -> None:
    st.markdown("### Dataset not loaded — precomputed EDA is still available")
    st.caption(
        "These figures were produced by `src/02_eda.py` from the full 360,000-row "
        "built-up sample. Load a CSV above to explore interactively."
    )
    pairs = [
        ("lst_distribution.png", "LST distribution"),
        ("lst_by_year.png", "Mean LST by year"),
        ("ndvi_vs_lst.png", "NDVI vs LST"),
        ("lst_by_lulc.png", "LST by land-cover class"),
        ("correlation_matrix.png", "Feature correlation"),
    ]
    cols = st.columns(2)
    for i, (name, caption) in enumerate(pairs):
        path = c.FIGS / name
        if path.exists():
            with cols[i % 2]:
                st.image(str(path), caption=caption, use_container_width=True)


def _filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        filtered = df

        with col1:
            if "YEAR" in df.columns:
                years = sorted(pd.to_numeric(df["YEAR"], errors="coerce").dropna().unique())
                picked = st.multiselect("Year", [int(y) for y in years], default=[int(y) for y in years])
                if picked:
                    filtered = filtered[filtered["YEAR"].isin(picked)]

        with col2:
            if "LULC" in df.columns:
                lulc = sorted(pd.to_numeric(df["LULC"], errors="coerce").dropna().unique())
                names = {int(v): c.LULC_NAMES.get(int(v), str(int(v))) for v in lulc}
                picked = st.multiselect(
                    "Land cover",
                    list(names),
                    default=list(names),
                    format_func=lambda v: names.get(v, str(v)),
                )
                if picked:
                    filtered = filtered[filtered["LULC"].isin(picked)]

        with col3:
            if "LST" in df.columns:
                lst = pd.to_numeric(filtered["LST"], errors="coerce").dropna()
                if not lst.empty:
                    lo, hi = float(lst.min()), float(lst.max())
                    if lo < hi:
                        bounds = st.slider(
                            "LST range (°C)",
                            min_value=round(lo, 1),
                            max_value=round(hi, 1),
                            value=(round(lo, 1), round(hi, 1)),
                            step=0.1,
                        )
                        filtered = filtered[
                            pd.to_numeric(filtered["LST"], errors="coerce").between(*bounds)
                        ]

    return filtered


def _dataset_view(df: pd.DataFrame) -> None:
    source = c.get_dataset_source()
    st.success(f"**{len(df):,} rows × {df.shape[1]} columns** loaded from {source}")
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("Clear data", use_container_width=True):
            c.clear_dataset()
            st.rerun()

    filtered = _filters(df)
    if filtered.empty:
        st.warning("No rows match the current filters.")
        return

    kpis = st.columns(5)
    kpis[0].metric("Rows", c.fmt_int(len(filtered)))
    if "LST" in filtered.columns:
        kpis[1].metric("Mean LST", f"{pd.to_numeric(filtered['LST'], errors='coerce').mean():.2f} °C")
    if "NDVI" in filtered.columns:
        kpis[2].metric("Mean NDVI", f"{pd.to_numeric(filtered['NDVI'], errors='coerce').mean():.3f}")
    if "NDBI" in filtered.columns:
        kpis[3].metric("Mean NDBI", f"{pd.to_numeric(filtered['NDBI'], errors='coerce').mean():.3f}")
    if "YEAR" in filtered.columns:
        kpis[4].metric("Years", c.fmt_int(filtered["YEAR"].nunique()))

    st.write("")
    tab_dist, tab_rel, tab_map, tab_table = st.tabs(
        ["Distributions", "Relationships", "Map", "Table & download"]
    )

    with tab_dist:
        numeric = [col for col in NUMERIC_HINTS if col in filtered.columns]
        if numeric:
            default = numeric.index("LST") if "LST" in numeric else 0
            col = st.selectbox("Variable", numeric, index=default, key="eda_hist_col")
            _histogram(filtered, col)
        if "YEAR" in filtered.columns and "LST" in filtered.columns:
            st.markdown("**Mean LST by year**")
            yearly = (
                filtered.assign(
                    LST=pd.to_numeric(filtered["LST"], errors="coerce"),
                    YEAR=pd.to_numeric(filtered["YEAR"], errors="coerce"),
                )
                .groupby("YEAR", as_index=False)["LST"]
                .mean()
            )
            yearly["Year"] = yearly["YEAR"].astype(int).astype(str)
            st.line_chart(yearly, x="Year", y="LST", height=260)
        if "LULC" in filtered.columns and "LST" in filtered.columns:
            st.markdown("**Mean LST by land-cover class**")
            by_lulc = (
                filtered.assign(
                    LST=pd.to_numeric(filtered["LST"], errors="coerce"),
                    LULC=pd.to_numeric(filtered["LULC"], errors="coerce"),
                )
                .groupby("LULC", as_index=False)["LST"]
                .mean()
            )
            by_lulc["Class"] = by_lulc["LULC"].map(
                lambda v: c.LULC_NAMES.get(int(v), str(int(v))) if pd.notna(v) else "—"
            )
            st.bar_chart(by_lulc, x="Class", y="LST", height=320)

    with tab_rel:
        numeric = [
            col
            for col in NUMERIC_HINTS
            if col in filtered.columns and pd.to_numeric(filtered[col], errors="coerce").notna().any()
        ]
        if len(numeric) >= 2:
            col1, col2 = st.columns(2)
            x = col1.selectbox("X axis", numeric, index=0, key="eda_x")
            y_default = numeric.index("LST") if "LST" in numeric else 1
            y = col2.selectbox("Y axis", numeric, index=y_default, key="eda_y")
            _scatter(filtered, x, y)
            st.markdown("**Correlation matrix**")
            _correlation(filtered)
        else:
            st.info("Not enough numeric columns.")

    with tab_map:
        if {"LON", "LAT"}.issubset(filtered.columns):
            color_options = [
                col
                for col in ["LST", "LST_ANOM", "LST_LOCAL", "NDVI", "NDBI", "BUILT_FRAC"]
                if col in filtered.columns
            ]
            color = st.selectbox("Colour by", color_options or [None], key="eda_map_colour")
            sample = c.sample_frame(filtered, 12_000)
            lat = pd.to_numeric(sample["LAT"], errors="coerce")
            lon = pd.to_numeric(sample["LON"], errors="coerce")
            sample = sample.assign(_lat=lat, _lon=lon).dropna(subset=["_lat", "_lon"])
            if sample.empty:
                st.info("No valid coordinates.")
            else:
                scatter = (
                    alt.Chart(sample)
                    .mark_circle(size=11, opacity=0.55)
                    .encode(
                        x=alt.X("_lon:Q", title="Longitude", scale=alt.Scale(zero=False)),
                        y=alt.Y("_lat:Q", title="Latitude", scale=alt.Scale(zero=False)),
                        color=alt.Color(f"{color}:Q", scale=alt.Scale(scheme="inferno"))
                        if color
                        else alt.value("#f97316"),
                        tooltip=[
                            alt.Tooltip("_lon:Q", title="Lon", format=".5f"),
                            alt.Tooltip("_lat:Q", title="Lat", format=".5f"),
                            *(
                                [alt.Tooltip(f"{color}:Q", title=color, format=".2f")]
                                if color
                                else []
                            ),
                        ],
                    )
                    .properties(height=600)
                )
                st.altair_chart(scatter, use_container_width=True)
                st.caption(
                    f"{len(sample):,} of {len(filtered):,} pixels. This is a point cloud in "
                    "geographic coordinates, not a tiled basemap; use **Heat maps** for the "
                    "published figures."
                )
        else:
            st.info("This sample has no LON/LAT columns, so it cannot be mapped.")

    with tab_table:
        st.markdown("**Summary statistics**")
        numeric = [col for col in NUMERIC_HINTS if col in filtered.columns]
        if numeric:
            summary = (
                filtered[numeric]
                .apply(pd.to_numeric, errors="coerce")
                .describe()
                .T.round(3)
                .reset_index()
                .rename(columns={"index": "Column"})
            )
            st.dataframe(summary, hide_index=True, use_container_width=True)
        st.markdown("**Preview (first 5,000 rows)**")
        st.dataframe(filtered.head(5000), hide_index=True, use_container_width=True, height=420)
        st.download_button(
            "Download filtered CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="coolkarachi_filtered.csv",
            mime="text/csv",
        )


def render() -> None:
    c.hero(
        "Interactive data lab",
        "Data",
        "Explorer",
        "Load the Earth Engine exports, filter by year and land cover, inspect "
        "distributions and correlations, and map the raw sample.",
    )

    _load_panel()
    st.divider()

    df = c.get_dataset()
    if df is None:
        _eda_gallery()
        with st.expander("Expected column schema"):
            st.markdown(
                "| Column | Meaning |\n"
                "|---|---|\n"
                "| `LST` | Land surface temperature (°C) |\n"
                "| `NDVI`, `NDBI`, `NDWI` | Vegetation, built-up and water indices |\n"
                "| `ALBEDO` | Broadband surface albedo (clipped 0.02–0.65) |\n"
                "| `BUILT_FRAC`, `TREE_FRAC`, `WATER_FRAC` | Dynamic World fractions (0–1) |\n"
                "| `ELEV` | Elevation (m, SRTM) |\n"
                "| `POP` | Persons per 100 m cell (WorldPop) |\n"
                "| `LON`, `LAT` | Pixel centroid (decimal degrees) |\n"
                "| `YEAR`, `LULC` | Observation year and dominant land-cover class |"
            )
    else:
        _dataset_view(df)
