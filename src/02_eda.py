"""Exploratory data analysis for the Karachi UHI dataset.

Expects the exported CSVs in data/ (karachi_features_<year>.csv).
Run: python src/02_eda.py
Figures are written to figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

LULC_NAMES = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "flooded veg",
    4: "crop",
    5: "shrub/scrub",
    6: "built",
    7: "bare",
    8: "snow/ice",
}

FEATURES = [
    "NDVI",
    "NDBI",
    "NDWI",
    "ALBEDO",
    "BUILT_FRAC",
    "TREE_FRAC",
    "WATER_FRAC",
    "ELEV",
    "POP",
]


def load(prefix: str) -> pd.DataFrame:
    files = sorted(DATA.glob(f"{prefix}_*.csv"))
    if not files:
        raise SystemExit(
            f"No CSVs matching {prefix}_*.csv found in {DATA}. Download them "
            "from the Google Drive folder 'karachi_uhi' first."
        )

    frames = []
    for path in files:
        df = pd.read_csv(path)
        frames.append(df)
        print(f"loaded {path.name}: {df.shape[0]:,} rows")

    df = pd.concat(frames, ignore_index=True)

    # Drop Earth Engine helper columns.
    drop = [c for c in df.columns if c.startswith(".") or c == "system:index"]
    df = df.drop(columns=drop, errors="ignore")

    df = df.dropna(subset=["LST"])
    df = df.drop_duplicates()
    print(f"\ncombined: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print("years:", sorted(df["YEAR"].unique().tolist()))
    return df


def summarise(df: pd.DataFrame) -> None:
    print("\n--- describe ---")
    print(df[["LST", *FEATURES]].describe().T.round(3).to_string())

    print("\n--- missing values (%) ---")
    miss = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
    print(miss[miss > 0].to_string())

    print("\n--- mean LST by year (C) ---")
    print(df.groupby("YEAR")["LST"].agg(["mean", "std", "count"]).round(2).to_string())

    print("\n--- mean LST by land cover (C) ---")
    by_lulc = (
        df.groupby("LULC")["LST"]
        .agg(["mean", "count"])
        .rename(index=LULC_NAMES)
        .round(2)
    )
    print(by_lulc.to_string())

    print("\n--- correlations with LST ---")
    print(
        df[["LST", *FEATURES]].corr()["LST"]
        .drop("LST")
        .sort_values(key=abs, ascending=False)
        .round(3)
        .to_string()
    )


def plots(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    # 1. LST distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["LST"], bins=60, kde=True, ax=ax, color="#c0392b")
    ax.set_title("Land surface temperature, Karachi (May–June)")
    ax.set_xlabel("LST (°C)")
    fig.tight_layout()
    fig.savefig(FIGS / "lst_distribution.png", dpi=200)
    plt.close(fig)

    # 2. Correlation matrix
    corr = df[["LST", *FEATURES]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Feature correlation")
    fig.tight_layout()
    fig.savefig(FIGS / "correlation_matrix.png", dpi=200)
    plt.close(fig)

    # 3. NDVI vs LST
    fig, ax = plt.subplots(figsize=(9, 6))
    hb = ax.hexbin(df["NDVI"], df["LST"], gridsize=50, cmap="magma", mincnt=1)
    fig.colorbar(hb, ax=ax, label="count")
    ax.set_xlabel("NDVI")
    ax.set_ylabel("LST (°C)")
    ax.set_title("Vegetation vs surface temperature")
    fig.tight_layout()
    fig.savefig(FIGS / "ndvi_vs_lst.png", dpi=200)
    plt.close(fig)

    # 4. LST by land-cover class
    fig, ax = plt.subplots(figsize=(11, 6))
    order = df.groupby("LULC")["LST"].median().sort_values().index
    sns.boxplot(
        data=df,
        x="LULC",
        y="LST",
        order=order,
        ax=ax,
        palette="viridis",
        showfliers=False,
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LULC_NAMES.get(int(i), str(i)) for i in order], rotation=30)
    ax.set_xlabel("")
    ax.set_ylabel("LST (°C)")
    ax.set_title("Surface temperature by land-cover class")
    fig.tight_layout()
    fig.savefig(FIGS / "lst_by_lulc.png", dpi=200)
    plt.close(fig)

    # 5. Yearly mean LST
    fig, ax = plt.subplots(figsize=(9, 5))
    yearly = df.groupby("YEAR")["LST"].mean()
    ax.plot(yearly.index, yearly.values, marker="o", linewidth=2, color="#c0392b")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean LST (°C)")
    ax.set_title("Annual mean surface temperature (May–June)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "lst_by_year.png", dpi=200)
    plt.close(fig)

    print(f"\nfigures written to {FIGS}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["urban", "full"],
        default="urban",
        help="urban = built-up sample (model training), full = all land cover.",
    )
    args = parser.parse_args()

    prefix = "karachi_ml" if args.source == "urban" else "karachi_features"
    df = load(prefix)
    summarise(df)
    plots(df)

    name = "karachi_urban_clean.csv" if args.source == "urban" else "karachi_full_clean.csv"
    clean = DATA / name
    df.to_csv(clean, index=False)
    print(f"clean dataset saved: {clean}")


if __name__ == "__main__":
    main()
