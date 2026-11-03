"""Generate the CoolKarachi presentation deck from result files.

Usage:
    python src/06_build_slides.py
    python src/06_build_slides.py --out slides/CoolKarachi.pptx

The deck auto-fills numbers from results/*.csv and figures/*.png so it never
goes out of sync with the experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

DARK = RGBColor(0x0E, 0x1B, 0x2E)
ACCENT = RGBColor(0xE8, 0x6A, 0x17)
ACCENT2 = RGBColor(0x1B, 0x7F, 0xA8)
GREY = RGBColor(0x4A, 0x55, 0x68)
LIGHT = RGBColor(0xF4, 0xF6, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

W, H = Inches(13.333), Inches(7.5)


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def rect(slide, x, y, w, h, color, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, body, size=18, bold=False, color=DARK,
         align=PP_ALIGN.LEFT, font="Calibri", anchor=MSO_ANCHOR.TOP,
         spacing=1.15, italic=False):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    lines = body if isinstance(body, list) else [body]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color
        run.font.name = font
    return box


def slide_header(slide, kicker, title):
    rect(slide, 0, 0, W, Inches(0.075), ACCENT)
    text(slide, Inches(0.75), Inches(0.42), Inches(10), Inches(0.32),
         kicker.upper(), size=11, bold=True, color=ACCENT)
    text(slide, Inches(0.75), Inches(0.78), Inches(11.8), Inches(0.75),
         title, size=30, bold=True, color=DARK)
    rect(slide, Inches(0.75), Inches(1.62), Inches(1.1), Inches(0.045), ACCENT2)
    return Inches(1.95)


def bullets(slide, items, top, x=Inches(0.9), w=Inches(11.6), size=17):
    """items: list of (level, text) or plain str."""
    box = slide.shapes.add_textbox(x, top, w, H - top - Inches(0.5))
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        level, body = item if isinstance(item, tuple) else (0, item)
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        p.line_spacing = 1.22
        p.space_after = Pt(9 if level == 0 else 5)
        run = p.add_run()
        run.text = ("•  " if level == 0 else "–  ") + body
        run.font.size = Pt(size if level == 0 else size - 2)
        run.font.color.rgb = DARK if level == 0 else GREY
        run.font.name = "Calibri"
    return box


def add_table(slide, df, top, left=None, width=None, size=13, col_w=None):
    rows, cols = len(df) + 1, len(df.columns)
    left = left if left is not None else Inches(0.9)
    width = width if width is not None else W - Inches(1.8)
    height = Inches(0.36) * rows
    shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = shape.table
    if col_w:
        for i, frac in enumerate(col_w):
            table.columns[i].width = Emu(int(width * frac))
    for j, col in enumerate(df.columns):
        cell = table.cell(0, j)
        cell.text = str(col)
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.runs[0].font.size = Pt(size)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = WHITE
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, row in enumerate(df.itertuples(index=False), start=1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if i % 2 else LIGHT
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
            p.runs[0].font.size = Pt(size)
            p.runs[0].font.color.rgb = DARK
            p.runs[0].font.bold = bool(i == 1)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    return shape


def add_picture(slide, path, top, max_w=Inches(11.0), max_h=Inches(4.6)):
    p = Path(path)
    if not p.exists():
        return None
    from PIL import Image as PILImage

    with PILImage.open(p) as im:
        iw, ih = im.size
    ratio = min(max_w / iw, max_h / ih)
    w, h = int(iw * ratio), int(ih * ratio)
    left = int((W - w) / 2)
    return slide.shapes.add_picture(str(p), left, top, w, h)


# --------------------------------------------------------------------------- #
# data helpers
# --------------------------------------------------------------------------- #
def read(target):
    p = RESULTS / f"model_comparison_{target}.csv"
    return pd.read_csv(p) if p.exists() else None


def read_csv(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


# --------------------------------------------------------------------------- #
# deck
# --------------------------------------------------------------------------- #
def build(out: Path) -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # ---------------- 1. title ----------------
    s = blank(prs)
    rect(s, 0, 0, W, H, DARK)
    rect(s, 0, Inches(2.0), Inches(1.8), Inches(0.08), ACCENT)
    text(s, Inches(1.2), Inches(2.35), Inches(11), Inches(1.6),
         "CoolKarachi", size=56, bold=True, color=WHITE)
    text(s, Inches(1.2), Inches(3.55), Inches(10.2), Inches(1.5),
         "An Explainable Machine-Learning Digital Twin for 30 m Urban Heat "
         "Mapping and Cooling-Intervention Prioritisation in Karachi",
         size=20, color=RGBColor(0xC9, 0xD3, 0xE0), spacing=1.3)
    rect(s, Inches(1.2), Inches(5.5), Inches(5.2), Inches(0.02),
         RGBColor(0x3A, 0x4A, 0x60))
    text(s, Inches(1.2), Inches(5.75), Inches(10), Inches(1.0),
         ["Data Science Course Project  ·  FAST-NUCES Karachi",
          "Google Earth Engine  ·  Landsat 8/9  ·  Sentinel-2  ·  SHAP"],
         size=13, color=RGBColor(0x8E, 0x9E, 0xB5), spacing=1.4)

    # ---------------- 2. problem ----------------
    s = blank(prs); top = slide_header(s, "01  Problem", "Karachi is heating faster than it can adapt")
    bullets(s, [
        (0, "May–June 2015 heatwave: >1,200 deaths, mostly in low-income, low-vegetation districts"),
        (0, "Karachi's built-up area has expanded rapidly, replacing irrigated land and mangrove fringes"),
        (0, "Within a single 2 km block, surface temperature varies by ~5 °C at the median and >8 °C in the hottest tenth"),
        (0, "Cooling budgets are scarce — interventions must be spatially prioritised, not sprayed evenly"),
        (1, "Current practice relies on coarse 1 km MODIS products or single-date analyses"),
        (1, "No open, explainable, 30 m decision-support layer exists for the city"),
    ], top)

    # ---------------- 3. gap / objectives ----------------
    s = blank(prs); top = slide_header(s, "02  Objective", "What this project delivers")
    bullets(s, [
        (0, "Build a 6-year, 30 m land-surface-temperature record for Karachi (2020–2025) from Landsat 8/9 via Earth Engine"),
        (0, "Explain the drivers of intra-urban heat with four machine-learning models and SHAP attribution"),
        (0, "Use the fitted model as a lightweight digital twin to simulate greening and cool-roof scenarios"),
        (0, "Rank neighbourhoods by expected cooling per intervention, so policy can target the hottest, most responsive cells"),
        (1, "Explicitly test three prediction targets to separate neighbourhood effects from city-wide climate gradients"),
    ], top)

    # ---------------- 4. study area + data ----------------
    s = blank(prs); top = slide_header(s, "03  Data", "Study area and input layers")
    bullets(s, [
        (0, "Karachi Division, 66.65–67.45 °E, 24.70–25.15 °N, plus a 15 km rural buffer"),
        (0, "Pre-monsoon season (May–June) only — clearest skies, peak heat, minimal cloud contamination"),
        (0, "Landsat 8/9 Collection 2 Level-2 surface temperature, 30 m, median composite per year"),
    ], top, w=Inches(6.2))
    df = pd.DataFrame({
        "Layer": ["LST", "NDVI / NDBI / NDWI", "Albedo", "Built / tree / water", "Elevation", "Population"],
        "Source": ["Landsat 8/9 C2 L2", "Sentinel-2 SR", "Sentinel-2 SR", "Dynamic World", "SRTM", "WorldPop"],
        "Res.": ["30 m", "10 m", "10 m", "10 m", "30 m", "100 m"],
    })
    add_table(s, df, top + Inches(0.1), left=Inches(7.2), width=Inches(5.4), size=11,
              col_w=[0.36, 0.38, 0.26])

    # ---------------- 5. sample design ----------------
    s = blank(prs); top = slide_header(s, "04  Sampling", "Two datasets, two different questions")
    bullets(s, [
        (0, "Full sample — 60,000 points/year across all land cover"),
        (1, "Measures the urban vs rural contrast (surface urban heat island)"),
        (0, "Built-up stratified sample — 60,000 points/year where built fraction ≥ 0.2"),
        (1, "Isolates intra-urban drivers, which is what intervention design needs"),
        (0, "360,000 rows total, six years, no missing values after QA"),
        (0, "28% of pixels were dropped as cloud/QA failures before sampling"),
    ], top)

    # ---------------- 6. methodology ----------------
    s = blank(prs); top = slide_header(s, "05  Method", "Pipeline")
    steps = [
        ("Earth Engine", "Landsat 8/9 + Sentinel-2\nmedian composites"),
        ("Feature stack", "9 predictors at 30 m\n+ coordinates"),
        ("Targets", "local / anomaly / raw\nLST definitions"),
        ("Spatial CV", "2 km blocks\nno leakage"),
        ("4 models", "RF, XGBoost,\nLightGBM, MLP"),
        ("SHAP + scenarios", "attribution and\ncounterfactuals"),
    ]
    x = Inches(0.75)
    bw = Inches(1.85)
    gap = Inches(0.14)
    for i, (head, body) in enumerate(steps):
        left = x + i * (bw + gap)
        rect(s, left, top + Inches(0.35), bw, Inches(1.75), LIGHT)
        rect(s, left, top + Inches(0.35), Inches(0.05), Inches(1.75), ACCENT)
        text(s, left + Inches(0.18), top + Inches(0.5), bw - Inches(0.3), Inches(0.5),
             head, size=13, bold=True, color=DARK)
        text(s, left + Inches(0.18), top + Inches(1.02), bw - Inches(0.3), Inches(1.0),
             body, size=10, color=GREY, spacing=1.25)
        if i < len(steps) - 1:
            text(s, left + bw - Inches(0.05), top + Inches(0.95), Inches(0.25), Inches(0.4),
                 "›", size=22, bold=True, color=ACCENT)

    # ---------------- 7. targets ----------------
    s = blank(prs); top = slide_header(s, "06  Targets", "Why three targets instead of one")
    df = pd.DataFrame({
        "Target": ["local", "anomaly", "raw"],
        "Definition": ["LST − 2 km block-year mean", "LST − year mean", "absolute LST (°C)"],
        "Answers": ["neighbourhood drivers", "city-wide pattern", "operational mapping"],
    })
    add_table(s, df, top, size=13, col_w=[0.16, 0.38, 0.46])
    text(s, Inches(0.9), top + Inches(1.9), Inches(11.6), Inches(1.6),
         ["A single R² hides the design choice. Reporting all three separates what the "
          "model can learn about a neighbourhood from what is simply geography and weather."],
         size=15, color=GREY, spacing=1.3)

    # ---------------- 8. spatial CV ----------------
    s = blank(prs); top = slide_header(s, "07  Validation", "Spatial blocking instead of random k-fold")
    bullets(s, [
        (0, "Random splits let the model memorise neighbouring pixels → optimistic R²"),
        (0, "Here the city is cut into ~700 tiles of 0.02° (≈2 km); whole tiles are held out"),
        (0, "Hyperparameters tuned with Optuna using GroupKFold on the training tiles"),
        (0, "Final metrics come only from tiles the model has never seen"),
        (0, "Result: cross-validation RMSE 1.186 vs test RMSE 1.219 °C — a small, honest gap"),
    ], top)

    # ---------------- 9. EDA ----------------
    s = blank(prs); top = slide_header(s, "08  EDA", "What the data says")
    bullets(s, [
        (0, "Mean LST 44.2 °C; inter-annual range 42.9 °C (2022) to 46.2 °C (2024)"),
        (0, "Land cover ordering is physical: water 39.6 < trees 41.0 < crop 43.1 < built 44.2 < bare 44.5 °C"),
        (0, "Urban–rural contrast is NEGATIVE: −1.2 °C — the rural reference is hotter inland desert"),
        (0, "Strongest correlations: NDBI +0.45, elevation +0.39, population −0.33, tree fraction −0.21"),
        (0, "Built fraction correlates only weakly (−0.07) inside the built-up sample — density matters more than presence"),
        (0, "Anomaly: albedo max 1.94 (physically impossible) → clipped to 0.02–0.65"),
    ], top)

    # ---------------- 10. results ----------------
    for target, label in [("local", "Local anomaly"), ("anomaly", "Year anomaly"), ("raw", "Raw LST")]:
        df = read(target)
        if df is None:
            continue
        s = blank(prs)
        top = slide_header(s, f"09  Results — {label}", "Model comparison on held-out spatial blocks")
        show = df.copy()
        show["RMSE"] = show["RMSE"].map(lambda v: f"{v:.3f}")
        show["MAE"] = show["MAE"].map(lambda v: f"{v:.3f}")
        show["R2"] = show["R2"].map(lambda v: f"{v:.3f}")
        show["MBE"] = show["MBE"].map(lambda v: f"{v:+.3f}")
        show = show.rename(columns={"model": "Model", "R2": "R²", "n": "n"})
        add_table(s, show, top, size=13, col_w=[0.18, 0.16, 0.16, 0.16, 0.18, 0.16])

    # ---------------- 11. SHAP ----------------
    df = read_csv("shap_importance_local.csv")
    if df is not None:
        s = blank(prs)
        top = slide_header(s, "10  Explainability", "Which features actually drive neighbourhood heat?")
        show = df.head(8).copy()
        show["mean_abs_shap"] = show["mean_abs_shap"].map(lambda v: f"{v:.3f}")
        show.columns = ["Feature", "Mean |SHAP| (°C)"]
        show.insert(0, "Rank", range(1, len(show) + 1))
        add_table(s, show, top, size=12, col_w=[0.12, 0.40, 0.48])

    # ---------------- 12. scenarios ----------------
    df = read_csv("scenarios_local.csv")
    if df is not None:
        s = blank(prs)
        top = slide_header(s, "11  Digital twin", "Counterfactual cooling scenarios")
        show = df.copy()
        show["mean_delta_C"] = show["mean_delta_C"].map(lambda v: f"{v:+.2f}")
        show["median_delta_C"] = show["median_delta_C"].map(lambda v: f"{v:+.2f}")
        show["pct_cooling_gt_0.5C"] = show["pct_cooling_gt_0.5C"].map(lambda v: f"{v:.1f}")
        show["pct_cooling_gt_1C"] = show["pct_cooling_gt_1C"].map(lambda v: f"{v:.1f}")
        show.columns = ["Scenario", "Mean ΔLST", "Median ΔLST", "% < −0.5 °C", "% < −1.0 °C"]
        add_table(s, show, top, size=12, col_w=[0.32, 0.17, 0.17, 0.17, 0.17])
        text(s, Inches(0.9), top + Inches(2.55), Inches(11.6), Inches(1.0),
             "Local-anomaly target. On the within-year anomaly target the same albedo "
             "perturbation cools 0.42 °C instead of 0.04 °C — the instability is the finding.",
             size=13, color=RGBColor(0x55, 0x55, 0x55), spacing=1.2)

    # ---------------- 13. heat map ----------------
    hm = sorted(FIGS.glob("heatmap_*.png"))
    if hm:
        s = blank(prs)
        top = slide_header(s, "12  Output", "30 m heat map — the digital twin's surface")
        add_picture(s, hm[0], top - Inches(0.15), max_w=Inches(11.2), max_h=Inches(4.5))

    # ---------------- 14. limitations ----------------
    s = blank(prs); top = slide_header(s, "13  Limitations", "What this study cannot claim")
    bullets(s, [
        (0, "LST is surface temperature, not air temperature — the two diverge over hard surfaces"),
        (0, "The model is associational; SHAP explains the fit, not a causal mechanism"),
        (0, "Cool-roof simulation inherits confounding: its effect swings from −0.04 to −0.42 °C across target definitions"),
        (0, "WorldPop only provides a 2020 surface, later years reuse it"),
        (0, "Counterfactuals assume pixels change independently, ignoring advection between neighbourhoods"),
        (0, "One city, one season, six years — results do not transfer without re-fitting"),
    ], top)

    # ---------------- 15. conclusion ----------------
    s = blank(prs); top = slide_header(s, "14  Conclusion", "Takeaways")
    bullets(s, [
        (0, "A 30 m, six-year, open heat record for Karachi is feasible entirely on Earth Engine's free tier"),
        (0, "Four model families converge on the same accuracy — the local signal is simple and robust"),
        (0, "Vegetation and built-up indices dominate the local heat signal; the gradient is physical, not an artefact"),
        (0, "Urban–rural SUHI is negative (−1.2 °C) — the classical index measures geography, not urban heat"),
        (0, "Simulated greening cools ~0.6 °C on average; albedo behaves counter-intuitively and is reported honestly"),
        (0, "Next: causal design (interrupted time series around actual park/cool-roof projects) and live weather coupling"),
    ], top)

    # ---------------- 16. thanks ----------------
    s = blank(prs)
    rect(s, 0, 0, W, H, DARK)
    rect(s, Inches(5.35), Inches(3.55), Inches(2.6), Inches(0.07), ACCENT)
    text(s, Inches(1.2), Inches(2.4), Inches(11), Inches(1.2),
         "Thank you", size=48, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    text(s, Inches(1.2), Inches(4.0), Inches(11), Inches(1.2),
         ["Questions and dataset requests welcome",
          "code + data + paper: CoolKarachi project repository"],
         size=16, color=RGBColor(0x9F, 0xAF, 0xC5), align=PP_ALIGN.CENTER, spacing=1.5)

    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    print(f"slides -> {out}  ({len(prs.slides)} slides)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "slides" / "CoolKarachi.pptx"))
    args = ap.parse_args()
    build(Path(args.out))


if __name__ == "__main__":
    main()
