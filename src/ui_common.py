"""Shared helpers for the CoolKarachi Streamlit dashboard.

The dashboard is deliberately tolerant of missing artefacts: the CSV exports
(~197 MB) and the fitted ``*.joblib`` models are git-ignored, so every page must
still render from the committed figures/results.  Data-dependent pages ask the
user to point the app at a local ``data/`` folder or upload CSVs.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
SLIDES = ROOT / "slides"
DOCS = ROOT / "docs"

# Keep these in sync with 03_models.py / 04_paper_assets.py.
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
TARGETS = {"local": "LST_LOCAL", "anomaly": "LST_ANOM", "raw": "LST"}
TARGET_LABELS = {
    "local": "Local anomaly (LST − 2 km block-year mean)",
    "anomaly": "Within-year anomaly (LST − year mean)",
    "raw": "Raw LST (°C)",
}
TARGET_SHORT = {
    "local": "Local anomaly",
    "anomaly": "Within-year anomaly",
    "raw": "Raw LST",
}
LIMITS = {"NDVI": (0.0, 0.9), "TREE_FRAC": (0.0, 0.6), "ALBEDO": (0.02, 0.65)}
LULC_NAMES = {
    0: "Water",
    1: "Trees",
    2: "Grass",
    3: "Flooded vegetation",
    4: "Crops",
    5: "Shrub / scrub",
    6: "Built-up",
    7: "Bare ground",
    8: "Snow / ice",
}

METRIC_HELP = {
    "RMSE": "Root-mean-square error in °C on held-out 2 km blocks.",
    "MAE": "Mean absolute error in °C.",
    "R2": "Coefficient of determination against the held-out blocks.",
    "MBE": "Mean bias error (prediction − observation), °C.",
}

SCENARIO_LABELS = {
    "greening_+10%_NDVI": "Greening (+0.10 NDVI, +0.05 tree fraction)",
    "cool_roof_+0.15_albedo": "Cool roof (+0.15 albedo)",
    "combined": "Combined greening + cool roof",
}


# --------------------------------------------------------------------------- #
# styling
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .ck-hero {
            border: 1px solid rgba(249, 115, 22, 0.28);
            background:
                radial-gradient(900px 280px at 8% 0%, rgba(251, 191, 36, 0.18), transparent 62%),
                linear-gradient(135deg, rgba(127, 29, 29, 0.16), rgba(15, 118, 110, 0.10));
            border-radius: 18px;
            padding: 1.6rem 1.8rem 1.45rem;
            margin-bottom: 1.1rem;
        }
        .ck-hero .ck-kicker {
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.72rem;
            font-weight: 700;
            color: #f97316;
            margin-bottom: 0.35rem;
        }
        .ck-hero h1 {
            font-size: 2.25rem;
            line-height: 1.1;
            margin: 0 0 0.5rem;
            padding: 0;
        }
        .ck-hero h1 span {
            background: linear-gradient(90deg, #f97316, #fbbf24 55%, #0f766e);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .ck-hero p {
            margin: 0;
            max-width: 68rem;
            opacity: 0.86;
            font-size: 0.98rem;
        }
        .ck-section {
            font-size: 0.74rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 700;
            color: #f97316;
            margin: 0.2rem 0 0.35rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 14px;
            padding: 0.55rem 0.65rem;
            background: rgba(249, 115, 22, 0.055);
        }
        [data-testid="stMetricLabel"] { opacity: 0.82; }
        [data-testid="stMetricLabel"] p {
            font-size: 0.72rem;
            line-height: 1.2;
            white-space: normal;
        }
        [data-testid="stMetricValue"] { font-size: 1.3rem; }
        .ck-note {
            border-left: 4px solid #0f766e;
            background: rgba(15, 118, 110, 0.08);
            border-radius: 0 10px 10px 0;
            padding: 0.8rem 1rem;
            font-size: 0.93rem;
        }
        .ck-warn {
            border-left: 4px solid #f97316;
            background: rgba(249, 115, 22, 0.08);
            border-radius: 0 10px 10px 0;
            padding: 0.8rem 1rem;
            font-size: 0.93rem;
        }
        .ck-pill {
            display: inline-block;
            padding: 0.12rem 0.55rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            border: 1px solid rgba(128, 128, 128, 0.35);
            margin: 0 0.25rem 0.3rem 0;
        }
        .ck-pill.ok { color: #0f766e; border-color: rgba(15, 118, 110, 0.5); }
        .ck-pill.missing { color: #b45309; border-color: rgba(180, 83, 9, 0.5); }
        div[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(kicker: str, title: str, accent: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="ck-hero">
          <div class="ck-kicker">{kicker}</div>
          <h1>{title} <span>{accent}</span></h1>
          <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(label: str) -> None:
    st.markdown(f'<div class="ck-section">{label}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="ck-note">{text}</div>', unsafe_allow_html=True)


def warning(text: str) -> None:
    st.markdown(f'<div class="ck-warn">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# files and small utilities
# --------------------------------------------------------------------------- #
def read_csv(path: Path) -> pd.DataFrame:
    """Read a committed result CSV, returning an empty frame if absent."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, max_entries=32)
def _read_csv_cached(path_str: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_csv(path_str)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return _read_csv_cached(str(path), path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False, max_entries=32)
def _read_upload_cached(name: str, payload: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(payload))


def read_uploaded(uploaded_files) -> pd.DataFrame:
    frames = []
    for f in uploaded_files:
        frames.append(_read_upload_cached(f.name, f.getvalue()))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def data_files() -> list[Path]:
    return sorted(DATA.glob("karachi_*.csv"))


def model_caches(target: str) -> list[Path]:
    found = sorted(RESULTS.glob(f"model_*_{target}.joblib"))
    quick = RESULTS / f"ui_surrogate_{target}.joblib"
    if quick.exists() and quick not in found:
        found.append(quick)
    return found


def available_targets() -> list[str]:
    return [t for t in ("local", "anomaly", "raw") if (RESULTS / f"model_comparison_{t}.csv").exists()]


def best_model(target: str = "local") -> pd.Series | None:
    df = load_csv(RESULTS / f"model_comparison_{target}.csv")
    if df.empty or "RMSE" not in df:
        return None
    return df.sort_values("RMSE").iloc[0]


def fmt_int(value) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "—"


def fmt_metric(value, digits: int = 2, signed: bool = False) -> str:
    try:
        return f"{float(value):+.{digits}f}" if signed else f"{float(value):.{digits}f}"
    except Exception:
        return "—"


def artifact_status() -> pd.DataFrame:
    rows = [
        ("Built-up ML exports", "data/karachi_ml_*.csv", len(list(DATA.glob("karachi_ml_*.csv")))),
        ("Full-cover exports", "data/karachi_features_*.csv", len(list(DATA.glob("karachi_features_*.csv")))),
        ("Cached models", "results/model_*.joblib", len(list(RESULTS.glob("model_*.joblib")))),
        ("Result tables", "results/*.csv", len(list(RESULTS.glob("*.csv")))),
        ("Figures", "figures/*.png", len(list(FIGS.glob("*.png")))),
        ("Paper", "paper/CoolKarachi.pdf", int((PAPER / "CoolKarachi.pdf").exists())),
        ("Slides", "slides/CoolKarachi.pptx", int((SLIDES / "CoolKarachi.pptx").exists())),
    ]
    return pd.DataFrame(rows, columns=["Artefact", "Path", "Found"])


def status_pills() -> None:
    data_n = len(data_files())
    model_n = len(list(RESULTS.glob("model_*.joblib")))
    pills = [
        f'<span class="ck-pill {"ok" if data_n else "missing"}">data: {data_n} file(s)</span>',
        f'<span class="ck-pill {"ok" if model_n else "missing"}">models: {model_n}</span>',
        '<span class="ck-pill ok">results: ready</span>',
        '<span class="ck-pill ok">figures: ready</span>',
    ]
    st.markdown("".join(pills), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# dataset preparation (mirrors 03_models.py)
# --------------------------------------------------------------------------- #
def prepare_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean and derive the columns used by the models and the dashboard."""
    df = raw.copy()

    drop = [c for c in df.columns if str(c).startswith(".") or c == "system:index"]
    df = df.drop(columns=drop, errors="ignore")

    if "LST" in df.columns:
        df = df.dropna(subset=["LST"])
    df = df.drop_duplicates()

    if "ALBEDO" in df.columns:
        df["ALBEDO"] = pd.to_numeric(df["ALBEDO"], errors="coerce").clip(0.02, 0.65)
    if "POP" in df.columns:
        pop = pd.to_numeric(df["POP"], errors="coerce").clip(lower=0)
        df["POP_LOG"] = np.log1p(pop)

    for col in FEATURES:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                lo, hi = values.quantile([0.001, 0.999])
                df[col] = values.clip(lo, hi)

    if {"LST", "YEAR"}.issubset(df.columns):
        df["LST_ANOM"] = df["LST"] - df.groupby("YEAR")["LST"].transform("mean")

    if {"LON", "LAT"}.issubset(df.columns) and len(df):
        lat0, lon0 = df["LAT"].min(), df["LON"].min()
        df["BLOCK"] = (
            np.floor((df["LAT"] - lat0) / 0.02).astype(int) * 10_000
            + np.floor((df["LON"] - lon0) / 0.02).astype(int)
        )
        if {"LST", "YEAR"}.issubset(df.columns):
            df["LST_LOCAL"] = df["LST"] - df.groupby(["YEAR", "BLOCK"])["LST"].transform("mean")

    return df


def feature_list(target: str, df: pd.DataFrame | None = None) -> list[str]:
    features = FEATURES + (["YEAR"] if target == "raw" else [])
    if df is None:
        return features
    return [f for f in features if f in df.columns]


def dataset_ready(target: str, df: pd.DataFrame) -> tuple[bool, str]:
    if df is None or df.empty:
        return False, "No dataset loaded."
    cols = feature_list(target, df) + [TARGETS[target]]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return False, "Missing columns: " + ", ".join(missing)
    return True, ""


def store_dataset(df: pd.DataFrame, source: str) -> None:
    st.session_state["ck_dataset"] = df
    st.session_state["ck_dataset_source"] = source


def clear_dataset() -> None:
    for key in ("ck_dataset", "ck_dataset_source"):
        st.session_state.pop(key, None)
    for key in list(st.session_state):
        if key.startswith("ck_model_"):
            st.session_state.pop(key, None)


def get_dataset() -> pd.DataFrame | None:
    df = st.session_state.get("ck_dataset")
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return None


def get_dataset_source() -> str:
    return st.session_state.get("ck_dataset_source", "")


def sample_frame(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False, max_entries=8)
def _load_model_cached(path_str: str, mtime_ns: int):
    import joblib

    return joblib.load(path_str)


def load_model(path: Path):
    return _load_model_cached(str(path), path.stat().st_mtime_ns)


def train_quick_surrogate(
    df: pd.DataFrame,
    target: str,
    sample_rows: int = 120_000,
) -> tuple[object, list[str], int]:
    """Fit a small LightGBM surrogate for the interactive what-if panel.

    This is intentionally separate from the paper's tuned models: it is fast,
    uses a random (not spatially blocked) split, and exists only so the UI can
    respond to sliders without shipping a 200 MB dataset.
    """
    from lightgbm import LGBMRegressor

    prepared = prepare_dataset(df)
    ok, message = dataset_ready(target, prepared)
    if not ok:
        raise ValueError(message)

    features = feature_list(target, prepared)
    target_col = TARGETS[target]
    train = prepared.dropna(subset=features + [target_col])
    if len(train) < 200:
        raise ValueError("Not enough complete rows to train a surrogate (need at least 200).")

    train = sample_frame(train, sample_rows)
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=20,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    model.fit(train[features].to_numpy(dtype=float), train[target_col].to_numpy(dtype=float))
    return model, features, len(train)


def predict_delta(
    model,
    baseline: pd.DataFrame,
    features: list[str],
    deltas: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (baseline prediction, predicted change) for feature perturbations."""
    base = baseline[features].astype(float).to_numpy()
    changed = baseline[features].astype(float).copy()
    for col, delta in deltas.items():
        if col not in changed.columns:
            continue
        lower, upper = LIMITS.get(col, (-np.inf, np.inf))
        changed[col] = (changed[col] + delta).clip(lower, upper)

    pred_base = model.predict(base)
    pred_changed = model.predict(changed.to_numpy())
    return np.asarray(pred_base), np.asarray(pred_changed) - np.asarray(pred_base)


def scenario_metrics(delta: np.ndarray) -> dict:
    delta = np.asarray(delta)
    return {
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "pct_cool_05": float(np.mean(delta < -0.5) * 100),
        "pct_cool_10": float(np.mean(delta < -1.0) * 100),
        "n": int(delta.size),
    }


def save_surrogate(model, target: str) -> Path:
    import joblib

    path = RESULTS / f"ui_surrogate_{target}.joblib"
    joblib.dump(model, path, compress=3)
    return path


def pretty_params(target: str) -> pd.DataFrame:
    path = RESULTS / f"best_params_{target}.json"
    if not path.exists():
        return pd.DataFrame()
    try:
        params = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()
    rows = []
    for model, cfg in params.items():
        for key, value in cfg.items():
            rows.append({"Model": model, "Parameter": key, "Value": str(value)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #
def read_paper_markdown() -> str:
    path = PAPER / "paper.md"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2].lstrip()
    return text


def file_bytes(path: Path) -> bytes | None:
    if not path.exists():
        return None
    return path.read_bytes()
