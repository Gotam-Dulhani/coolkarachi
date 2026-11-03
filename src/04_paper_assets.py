"""Turn result CSVs into IEEE-ready tables and figures.

Run after 03_models.py has produced results for the targets you want.

Usage:
    python src/04_paper_assets.py                     # all available targets
    python src/04_paper_assets.py --targets local raw

Outputs
    paper/tables/*.tex           LaTeX (booktabs) tables
    figures/heatmap_*.png        observed vs predicted LST maps
    figures/suhi_annual.png      urban–rural contrast by year
    results/suhi_full_sample.csv per-year and pooled urban/rural LST
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
TABLES = ROOT / "paper" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

TARGET_LABEL = {
    "local": "Local anomaly (LST $-$ 2~km block mean)",
    "anomaly": "Within-year anomaly (LST $-$ year mean)",
    "raw": "Raw LST (°C)",
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
    "POP_LOG",
]


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def model_table(targets: list[str]) -> str:
    blocks = []
    for t in targets:
        path = RESULTS / f"model_comparison_{t}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        rows = "\n".join(
            f"{r.model} & {r.RMSE:.3f} & {r.MAE:.3f} & {r.R2:.3f} & "
            f"{r.MBE:+.3f} & {int(r.n):,} \\\\"
            for r in df.itertuples()
        )
        blocks.append(
            f"\\midrule\n\\multicolumn{{6}}{{l}}{{\\textit{{{TARGET_LABEL[t]}}}}} \\\\\n"
            f"\\midrule\n{rows}"
        )

    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Out-of-sample performance of the four models under spatial "
        "block cross-validation. Best value per target in bold.}\n"
        "\\label{tab:models}\n"
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        "Model & RMSE & MAE & $R^2$ & MBE & $n$ \\\\\n"
        + "\n".join(blocks)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def shap_table(targets: list[str]) -> str:
    frames = {}
    for t in targets:
        path = RESULTS / f"shap_importance_{t}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["rank"] = np.arange(1, len(df) + 1)
            frames[t] = df.set_index("feature")
    if not frames:
        return ""

    feats = frames[next(iter(frames))].index.tolist()
    header = " & ".join(f"\\multicolumn{{2}}{{c}}{{{t}}}" for t in frames)
    sub = " & ".join("rank & $|\\phi|$" for _ in frames)
    body = []
    for f in feats:
        cells = []
        for t in frames:
            if f in frames[t].index:
                r = frames[t].loc[f]
                cells.append(f"{int(r['rank'])} & {r['mean_abs_shap']:.3f}")
            else:
                cells.append("-- & --")
        body.append(f"{f} & " + " & ".join(cells) + " \\\\")

    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Global feature importance from SHAP (mean absolute SHAP "
        "value, °C). Rank 1 = most influential.}\n"
        "\\label{tab:shap}\n"
        f"\\begin{{tabular}}{{l{'cc' * len(frames)}}}\n\\toprule\n"
        f"& {header} \\\\\n& {sub} \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def scenario_table(targets: list[str]) -> str:
    blocks = []
    for t in targets:
        path = RESULTS / f"scenarios_{t}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        blocks.append(
            f"\\midrule\n\\multicolumn{{5}}{{l}}{{\\textit{{{TARGET_LABEL[t]}}}}} \\\\\n"
            "\\midrule\n"
            + "\n".join(
                f"{r.scenario.replace('_', ' ')} & {r.mean_delta_C:+.2f} & "
                f"{r.median_delta_C:+.2f} & {r.pct_cooling_gt_0_5C:.1f} & "
                f"{r.pct_cooling_gt_1C:.1f} \\\\"
                for r in df.rename(
                    columns={
                        "pct_cooling_gt_0.5C": "pct_cooling_gt_0_5C",
                        "pct_cooling_gt_1C": "pct_cooling_gt_1C",
                    }
                ).itertuples()
            )
        )

    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Counterfactual cooling potential of intervention scenarios "
        "(change in predicted LST).}\n"
        "\\label{tab:scenarios}\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Scenario & mean $\\Delta$LST & median $\\Delta$LST & "
        "$\\%$ cells $<-0.5$°C & $\\%$ cells $<-1.0$°C \\\\\n"
        + "\n".join(blocks)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# --------------------------------------------------------------------------- #
# heat maps
# --------------------------------------------------------------------------- #
def load_built() -> pd.DataFrame:
    files = sorted(DATA.glob("karachi_ml_*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = df.dropna(subset=["LST", "LON", "LAT"]).drop_duplicates()
    df["ALBEDO"] = df["ALBEDO"].clip(0.02, 0.65)
    df["POP_LOG"] = np.log1p(df["POP"].clip(lower=0))
    for col in FEATURES:
        lo, hi = df[col].quantile([0.001, 0.999])
        df[col] = df[col].clip(lo, hi)
    # 2 km blocks, identical to 03_models.py
    df["BLOCK"] = (
        np.floor((df["LAT"] - df["LAT"].min()) / 0.02).astype(int) * 10_000
        + np.floor((df["LON"] - df["LON"].min()) / 0.02).astype(int)
    )
    df["LST_LOCAL"] = df["LST"] - df.groupby(["YEAR", "BLOCK"])["LST"].transform("mean")
    return df


def heatmap(df: pd.DataFrame, year: int) -> None:
    sub = df[df["YEAR"] == year]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharex=True, sharey=True)
    vmin, vmax = 35, 55
    for ax, col, title in zip(
        axes, ["LST", "LST_LOCAL"], ["Observed LST (°C)", "Local anomaly (°C)"]
    ):
        if col not in sub:
            continue
        if col == "LST_LOCAL":
            vmin, vmax = -4, 4
        sc = ax.scatter(
            sub["LON"], sub["LAT"], c=sub[col], s=0.6, cmap="inferno",
            vmin=vmin, vmax=vmax, marker="s", linewidths=0,
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="°C")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_aspect("equal")
    axes[0].set_ylabel("Latitude")
    fig.suptitle(f"Karachi, pre-monsoon {year} (30 m Landsat LST)")
    fig.tight_layout()
    fig.savefig(FIGS / f"heatmap_{year}.png", dpi=220)
    plt.close(fig)


def md_models(target: str) -> str:
    df = pd.read_csv(RESULTS / f"model_comparison_{target}.csv")
    df = df.sort_values("RMSE").reset_index(drop=True)
    lines = ["| Model | RMSE (°C) | MAE (°C) | R² | MBE (°C) | n |",
             "|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(
            f"| {r.model} | {r.RMSE:.3f} | {r.MAE:.3f} | {r.R2:.3f} | "
            f"{r.MBE:+.3f} | {int(r.n):,} |"
        )
    return "\n".join(lines)


def md_shap(target: str) -> str:
    p = RESULTS / f"shap_importance_{target}.csv"
    if not p.exists():
        return ""
    df = pd.read_csv(p)
    lines = ["| Rank | Feature | Mean absolute SHAP (°C) |", "|---|---|---|"]
    for i, r in enumerate(df.itertuples(), start=1):
        lines.append(f"| {i} | {r.feature} | {r.mean_abs_shap:.3f} |")
    return "\n".join(lines)


def md_scenarios(target: str) -> str:
    if target == "all":
        labels = {"local": "Local", "anomaly": "Within-year", "raw": "Raw"}
        blocks = []
        for t in ("local", "anomaly", "raw"):
            body = md_scenarios(t)
            if not body:
                continue
            rows = body.splitlines()[2:]
            blocks.extend(f"| {labels[t]} {row}" for row in rows)
        if not blocks:
            return ""
        return "\n".join(
            [
                "| Target | Scenario | Mean ΔLST (°C) | Median ΔLST (°C) | Cells < −0.5 °C (%) | Cells < −1.0 °C (%) |",
                "|---|---|---|---|---|---|",
                *blocks,
            ]
        )

    p = RESULTS / f"scenarios_{target}.csv"
    if not p.exists():
        return ""
    df = pd.read_csv(p).rename(
        columns={
            "pct_cooling_gt_0.5C": "pct_cooling_gt_0_5C",
            "pct_cooling_gt_1C": "pct_cooling_gt_1C",
        }
    )
    lines = ["| Scenario | Mean ΔLST (°C) | Median ΔLST (°C) | Cells < −0.5 °C (%) | Cells < −1.0 °C (%) |",
             "|---|---|---|---|---|"]
    for r in df.itertuples():
        name = r.scenario.replace("_", " ")
        lines.append(
            f"| {name} | {r.mean_delta_C:+.2f} | {r.median_delta_C:+.2f} | "
            f"{r.pct_cooling_gt_0_5C:.1f} | {r.pct_cooling_gt_1C:.1f} |"
        )
    return "\n".join(lines)


def md_params(target: str) -> str:
    p = RESULTS / f"best_params_{target}.json"
    if not p.exists():
        return ""
    params = json.loads(p.read_text())
    lines = ["| Model | Selected hyperparameters |", "|---|---|"]
    for name, cfg in params.items():
        pretty = ", ".join(
            f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}"
            for k, v in cfg.items()
        )
        lines.append(f"| {name} | {pretty} |")
    return "\n".join(lines)


def inject(template: Path) -> None:
    """Replace <!-- results:key --> blocks with freshly generated markdown."""
    text = template.read_text(encoding="utf-8")

    def value(key: str) -> str:
        kind, _, target = key.partition(":")
        fn = {
            "models": md_models,
            "shap": md_shap,
            "scenarios": md_scenarios,
            "params": md_params,
        }.get(kind)
        if fn is None or not target:
            return ""
        try:
            return fn(target)
        except FileNotFoundError:
            return "_Not available._"

    pat = re.compile(r"<!--\s*results:([^\s>]+)\s*-->(.*?)<!--\s*/results:[^\s>]+\s*-->", re.S)
    new = pat.sub(lambda m: f"<!-- results:{m.group(1)} -->\n{value(m.group(1))}\n<!-- /results:{m.group(1)} -->", text)
    if new != text:
        template.write_text(new, encoding="utf-8")
        print(f"injected tables into {template}")
    else:
        print("no change (markers not found?)")


def suhi_summary() -> None:
    """Urban–rural surface heat island from the full all-land-cover sample."""
    files = sorted(DATA.glob("karachi_features_*.csv"))
    if not files:
        print("skipping SUHI: no karachi_features_*.csv in data/")
        return
    cols = ["LST", "BUILT_FRAC", "ELEV", "LULC", "YEAR"]
    df = pd.concat((pd.read_csv(f, usecols=cols) for f in files), ignore_index=True)
    df = df.dropna(subset=["LST", "BUILT_FRAC"])

    rows = []
    for year, d in df.groupby("YEAR"):
        urban = d.loc[d["BUILT_FRAC"] >= 0.2, "LST"]
        rural = d.loc[d["BUILT_FRAC"] < 0.1, "LST"]
        rows.append(
            {
                "period": str(int(year)),
                "n_urban": int(urban.size),
                "n_rural": int(rural.size),
                "lst_urban_C": float(urban.mean()),
                "lst_rural_C": float(rural.mean()),
                "suhi_C": float(urban.mean() - rural.mean()),
            }
        )

    urban = df.loc[df["BUILT_FRAC"] >= 0.2, "LST"]
    rural = df.loc[df["BUILT_FRAC"] < 0.1, "LST"]
    pooled = {
        "period": "pooled",
        "n_urban": int(urban.size),
        "n_rural": int(rural.size),
        "lst_urban_C": float(urban.mean()),
        "lst_rural_C": float(rural.mean()),
        "suhi_C": float(urban.mean() - rural.mean()),
    }
    out = pd.DataFrame(rows + [pooled])
    out.to_csv(RESULTS / "suhi_full_sample.csv", index=False)

    per_year = out[out["period"] != "pooled"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(per_year["period"], per_year["suhi_C"], color="#c0504d", width=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Urban − rural LST (°C)")
    ax.set_xlabel("Year")
    ax.set_title("Surface urban heat island, full sample")
    for x, v in zip(per_year["period"], per_year["suhi_C"]):
        ax.text(x, v - 0.08, f"{v:+.2f}", ha="center", va="top", fontsize=8)
    ax.set_ylim(min(per_year["suhi_C"]) - 0.45, 0.15)
    fig.tight_layout()
    fig.savefig(FIGS / "suhi_annual.png", dpi=200)
    plt.close(fig)

    print(
        f"SUHI (full sample, pooled): urban {pooled['lst_urban_C']:.2f} °C vs "
        f"rural {pooled['lst_rural_C']:.2f} °C -> {pooled['suhi_C']:+.2f} °C "
        f"[n={len(df):,}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets", nargs="+", default=["local", "anomaly", "raw"],
        choices=["local", "anomaly", "raw"],
    )
    parser.add_argument("--inject", metavar="PAPER_MD",
                        help="Fill <!-- results:* --> blocks in this markdown file.")
    parser.add_argument("--heatmaps", action="store_true", default=True,
                        help="Render per-year observed / anomaly heat maps.")
    parser.add_argument("--no-heatmaps", dest="heatmaps", action="store_false")
    args = parser.parse_args()

    if args.inject:
        inject(Path(args.inject))

    suhi_summary()

    available = [t for t in args.targets if (RESULTS / f"model_comparison_{t}.csv").exists()]
    missing = [t for t in args.targets if t not in available]
    if missing:
        print(f"skipping (no results yet): {', '.join(missing)}")
    if not available:
        raise SystemExit("No result files found. Run 03_models.py first.")

    written = []
    for name, fn in [
        ("models", model_table),
        ("shap", shap_table),
        ("scenarios", scenario_table),
    ]:
        tex = fn(available)
        if tex:
            path = TABLES / f"{name}.tex"
            path.write_text(tex, encoding="utf-8")
            written.append(path)

    if args.heatmaps:
        df = load_built()
        if not df.empty:
            for year in sorted(df["YEAR"].unique()):
                heatmap(df, int(year))
            print(f"heatmaps written for {df['YEAR'].nunique()} years")

    for p in written:
        print(f"table -> {p}")


if __name__ == "__main__":
    main()
