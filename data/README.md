# Dataset

The dataset is produced entirely from public archives through Google Earth Engine.
Nothing is hand-labelled and nothing is proprietary.

## Files

| File | Rows | Size | Purpose |
|---|---|---|---|
| `karachi_features_<year>.csv` | ~38,500 | ~8.6 MB | full sample across all land cover (urban–rural SUHI contrast) |
| `karachi_ml_<year>.csv` | 60,000 | ~13.6 MB | built-up stratified sample with coordinates (all ML experiments) |
| `karachi_urban_clean.csv` | 360,000 | ~64 MB | concatenated, cleaned built-up sample written by `02_eda.py` |

Years: 2020, 2021, 2022, 2023, 2024, 2025. Season: May–June (pre-monsoon).
`04_paper_assets.py` also writes `results/suhi_full_sample.csv`, the per-year
urban–rural contrast used in Section 5 of the paper.

## Columns

| Column | Unit | Source |
|---|---|---|
| `LST` | °C | Landsat 8/9 Collection 2 Level-2, `ST_B10` |
| `NDVI`, `NDBI`, `NDWI` | index | Sentinel-2 surface reflectance, May–June median |
| `ALBEDO` | 0–1 (raw up to 1.94, clipped to 0.02–0.65) | Sentinel-2 |
| `BUILT_FRAC`, `TREE_FRAC`, `WATER_FRAC` | 0–1 | Dynamic World class probabilities |
| `LULC` | class label | Dynamic World dominant class |
| `ELEV` | m | SRTM 1 arc-second |
| `POP` | persons per 100 m cell | WorldPop 2020 surface (reused 2021–2025) |
| `LON`, `LAT` | decimal degrees | `ee.Image.pixelLonLat()` |
| `YEAR` | — | observation year |

## How to regenerate

```powershell
python src/01_export_features.py --years 2020 2021 2022 2023 2024 2025
python src/01_export_features.py --years 2020 2021 2022 2023 2024 2025 --urban-only --no-raster
```

Exports are written to the Google Drive folder `karachi_uhi`; download them into
this directory. The archived copy used in the paper is at

https://drive.google.com/drive/folders/1ZbUw7r6Sor7C47uXzFwsOGu8laFZ32Cg

Total wall time is roughly one hour depending on queue position.

## Known data issues

- `ALBEDO` reaches 1.94 before cleaning, which is physically impossible and
  indicates residual cloud or a compositing artefact. It is clipped to
  0.02–0.65 in all models.
- `POP` is static across years because the WorldPop collection used provides a
  single 2020 surface.
- `ELEV` and `POP` are collinear with distance from the coast. Elevation
  dominance in the models is a coastal–inland climate gradient, not a lapse-rate
  effect.
- Duplicate 2024 exports may appear in Drive with a `(1)` suffix; they are
  identical and can be ignored.

## Licence and attribution

All inputs are open: Landsat (USGS), Sentinel-2 (ESA), Dynamic World (Google),
SRTM (NASA), WorldPop (University of Southampton). Cite the original providers
when redistributing.
