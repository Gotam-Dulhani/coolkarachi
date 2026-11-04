# Karachi Urban Heat Island — ML + GEE Digital Twin

Course project: **CoolKarachi — An Explainable Machine-Learning Digital Twin for 30 m Urban Heat Mapping and Cooling-Intervention Prioritisation in Karachi**

## Objective
1. Build a 30 m land surface temperature (LST) record for Karachi (Landsat 8/9, 2020–2025).
2. Explain what drives the surface urban heat island (SUHI) using ML + SHAP.
3. Simulate greening / cool-roof scenarios to prioritise cooling interventions.

## Dataset

**https://drive.google.com/drive/folders/1ZbUw7r6Sor7C47uXzFwsOGu8laFZ32Cg** — the twelve yearly exports plus the cleaned 360,000-row table (~197 MB). The CSVs are not committed to this repository; see Step 3.

## Repository layout
```
.
├── app.py                     # Streamlit dashboard entrypoint
├── src/
│   ├── 00_check.py            # verify Earth Engine connection
│   ├── 01_export_features.py  # build feature stack + export CSV samples to Drive
│   ├── 02_eda.py              # exploratory analysis + figures
│   ├── 03_models.py           # 4 models, Optuna, spatial CV, SHAP, scenarios
│   ├── 04_paper_assets.py     # LaTeX tables, heat maps, SUHI summary, table injection
│   ├── 05_build_paper.py      # paper/paper.md -> paper/CoolKarachi.pdf
│   ├── 06_build_slides.py     # slides/CoolKarachi.pptx
│   └── ui_*.py                # dashboard pages and shared helpers
├── data/                      # downloaded CSVs go here (not committed)
├── figures/                   # output plots
├── results/                   # metrics, SHAP, scenarios, cached models
├── paper/                     # paper.md, tables/*.tex, CoolKarachi.pdf
├── slides/                    # generated deck
├── notebooks/                 # scratch notebooks
├── requirements.txt
└── README.md
```

## Streamlit dashboard

A seven-page dashboard sits on top of the existing pipeline. It works from the
committed results/figures even when the large CSVs are absent, and unlocks live
exploration when the exports are available.

```powershell
# from the project root, with the same virtual environment
streamlit run app.py
```

Open `http://localhost:8501`.

| Page | What it does |
|---|---|
| **Overview** | Headline SUHI, model and intervention numbers, pipeline summary and artefact health |
| **Data explorer** | Load `data/karachi_*.csv` or upload CSVs, filter by year/land cover/LST, plot distributions, correlations and maps, download a filtered sample |
| **Model results** | Compare RF/XGB/LGBM/MLP across all three targets, inspect SHAP attribution and tuned hyperparameters |
| **Cooling scenarios** | Published counterfactual results plus a live NDVI/tree/albedo simulator |
| **Heat maps** | Per-year observed LST and local-anomaly maps, greening scenario map, SUHI chart |
| **Methodology** | Target definitions, feature dictionary, spatial validation protocol and limitations |
| **Paper & documents** | Render/download the paper, slides, reports and every generated figure |

Notes:

- The dashboard does **not** require Earth Engine authentication or a network call.
- Data-dependent pages show the precomputed EDA figures when no CSV is loaded.
- The live simulator uses a cached `results/model_*_<target>.joblib` if present;
  otherwise it can train a small LightGBM surrogate from the loaded sample and
  save it as `results/ui_surrogate_<target>.joblib`.
- The simulator is for orientation. The paper's spatially blocked models remain
  the validated results.
- If `streamlit` is missing, run `pip install -r requirements.txt` first.

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# one-time Google authentication (opens browser)
earthengine authenticate
```

If `earthengine` is not recognised, use:

```powershell
python -c "import ee; ee.Authenticate()"
```

## Step 1 — verify connection

```powershell
python src/00_check.py
```

Expected output: `Earth Engine OK` plus LST statistics for a point in Karachi.

## Step 2 — export features

```powershell
python src/01_export_features.py --years 2020 2021 2022 2023 2024 2025
```

This starts Earth Engine export tasks. Each year produces:

- `karachi_features_<year>.csv` — requests 60,000 points across all land cover, ~38,500 after masked pixels are dropped (urban–rural SUHI reference)
- `karachi_lst_<year>.tif` — LST raster for mapping

For the ML models we also export an **urban-only stratified sample**:

```powershell
python src/01_export_features.py --years 2020 2021 2022 2023 2024 2025 --urban-only --no-raster
```

- `karachi_ml_<year>.csv` — 60,000 built-up pixels per year
  (Dynamic World built fraction ≥ 0.2, stratified sampling), including
  `LON`/`LAT` so that spatial cross-validation and heat maps are possible

Why two datasets: the full sample measures **urban vs rural** contrast (SUHI), while the urban sample isolates **intra-urban** drivers for the ML models.

## Step 3 — download from Google Drive

The exported tables are archived here:

**https://drive.google.com/drive/folders/1ZbUw7r6Sor7C47uXzFwsOGu8laFZ32Cg**

Download the twelve yearly `karachi_features_*.csv` / `karachi_ml_*.csv` files
plus `karachi_urban_clean.csv` into `data/`.

> The CSVs are not committed to this repository because they total ~197 MB.

> Export tasks run in the background (minutes to ~1 hour). Check progress at
> `https://code.earthengine.google.com/tasks`.

## Feature dictionary

| Column | Meaning | Source |
|---|---|---|
| `LST` | Land surface temperature (°C) | Landsat 8/9 C2 L2 `ST_B10` |
| `NDVI` | Vegetation index | Sentinel-2 |
| `NDBI` | Built-up index | Sentinel-2 |
| `NDWI` | Water index | Sentinel-2 |
| `ALBEDO` | Surface albedo | Sentinel-2 |
| `BUILT_FRAC` | Built-up fraction (0–1) | Dynamic World |
| `TREE_FRAC` | Tree cover fraction (0–1) | Dynamic World |
| `WATER_FRAC` | Water fraction (0–1) | Dynamic World |
| `LULC` | Dominant land-cover class | Dynamic World |
| `ELEV` | Elevation (m) | SRTM |
| `POP` | Population per 100 m cell | WorldPop (2020 surface, reused for later years) |
| `LON` / `LAT` | Pixel centroid coordinates (decimal degrees) | `ee.Image.pixelLonLat()` |
| `YEAR` | Year of observation | — |

## Step 4 — exploratory analysis

```powershell
python src/02_eda.py --source urban    # built-up ML sample (writes karachi_urban_clean.csv)
```

`--source full` is optional; it runs the same EDA on the all-land-cover sample.
The urban–rural SUHI summary used in the paper is computed directly from the
`karachi_features_*.csv` files by `04_paper_assets.py` (Step 6).

## Step 5 — train and compare models

```powershell
python src/03_models.py --trials 15                  # local LST anomaly (default)
python src/03_models.py --target anomaly --trials 12 # within-year anomaly
python src/03_models.py --target raw --trials 12     # absolute LST (adds YEAR)

python src/03_models.py --skip-tuning                # reuse best_params_<target>.json
python src/03_models.py --explain-only               # reuse cached results/model_*.joblib
```

Models: Random Forest, XGBoost, LightGBM, MLP. Tuning with Optuna + `GroupKFold`
on 2 km spatial blocks; final evaluation on held-out blocks (no spatial leakage).

Fitted models are cached to `results/model_<name>_<target>.joblib`, so
`--explain-only` reruns SHAP, scenarios and figures without refitting. SHAP is
computed on LightGBM because a deep 600-tree forest is prohibitively slow to
explain and the four models are tied within 0.005 °C RMSE.

Outputs land in `results/` (`model_comparison_*.csv`, `best_params_*.json`,
`shap_importance_*.csv`, `scenarios_*.csv`) and `figures/`.

## Step 6 — paper tables, heat maps and PDF

```powershell
python src/04_paper_assets.py                  # LaTeX tables + per-year heat maps
python src/04_paper_assets.py --inject paper/paper.md   # fill results tables in the paper
python src/05_build_paper.py                   # paper/paper.md -> paper/CoolKarachi.pdf
python src/06_build_slides.py                  # slides/CoolKarachi.pptx
```

`04_paper_assets.py` writes `paper/tables/*.tex` and `figures/heatmap_<year>.png`.
`05_build_paper.py` renders a Markdown subset (headings, tables, figures,
bullets, numbered lists, references) to a Times-typeset A4 PDF.

## Notes

- Season: **May–June** (pre-monsoon, clearest skies and peak heat).
- LST is *surface* temperature, not air temperature — state this in the paper.
- The urban–rural surface heat island of Karachi is **negative** (−1.2 °C pooled,
  2020–2025): the rural reference is hotter inland desert. Hence the
  block-relative local anomaly is the primary modelling target.
- Landsat revisit is 16 days, so gaps are normal; median compositing handles them.
- Project ID: `karachi-uhi`
