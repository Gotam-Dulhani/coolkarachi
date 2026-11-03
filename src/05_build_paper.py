"""Build the research paper PDF from paper/paper.md.

A small Markdown subset is supported so the paper stays easy to edit:

    ---                 front matter (title / authors / affiliation)
    title: ...
    ---
    # Section
    ## Subsection
    plain paragraphs (auto justified)
    - bullet
    1. numbered
    | a | b |           pipe tables (2nd row must be the |---| separator)
    ![caption](figures/x.png){w=0.85}
    ---page---          explicit page break
    **bold**  *italic*  `mono`

Usage:
    python src/05_build_paper.py
    python src/05_build_paper.py --src paper/paper.md --out paper/CoolKarachi.pdf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[1]

PAGE_W, PAGE_H = A4
M_LEFT = M_RIGHT = 2.1 * cm
M_TOP = 2.0 * cm
M_BOTTOM = 2.0 * cm
CONTENT_W = PAGE_W - M_LEFT - M_RIGHT


# --------------------------------------------------------------------------- #
# fonts — real TrueType so Unicode (Δ, °, ≥, −) renders correctly
# --------------------------------------------------------------------------- #
FONTS = {
    "body": "Times-Roman",
    "bold": "Times-Bold",
    "italic": "Times-Italic",
    "bi": "Times-BoldItalic",
    "mono": "Courier",
}


def register_fonts() -> None:
    win = Path(r"C:\Windows\Fonts")
    files = {
        "body": "times.ttf",
        "bold": "timesbd.ttf",
        "italic": "timesi.ttf",
        "bi": "timesbi.ttf",
        "mono": "cour.ttf",
    }
    if not all((win / f).exists() for f in files.values()):
        return
    names = {
        "body": "TNR",
        "bold": "TNR-Bold",
        "italic": "TNR-Italic",
        "bi": "TNR-BoldItalic",
        "mono": "CourierTTF",
    }
    for key, fname in files.items():
        pdfmetrics.registerFont(TTFont(names[key], str(win / fname)))
    addMapping("TNR", 0, 0, "TNR")
    addMapping("TNR", 1, 0, "TNR-Bold")
    addMapping("TNR", 0, 1, "TNR-Italic")
    addMapping("TNR", 1, 1, "TNR-BoldItalic")
    FONTS.update(names)


# --------------------------------------------------------------------------- #
# styles
# --------------------------------------------------------------------------- #
def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName=FONTS["body"],
        fontSize=10,
        leading=13.4,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        firstLineIndent=0,
    )
    return {
        "title": ParagraphStyle(
            "TitleX", parent=body, fontName=FONTS["bold"], fontSize=19,
            leading=23, alignment=TA_CENTER, spaceAfter=10,
        ),
        "authors": ParagraphStyle(
            "Authors", parent=body, fontName=FONTS["body"], fontSize=11.5,
            leading=15, alignment=TA_CENTER, spaceAfter=2,
        ),
        "affil": ParagraphStyle(
            "Affil", parent=body, fontName=FONTS["italic"], fontSize=9.5,
            leading=12.5, alignment=TA_CENTER, spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "H1", parent=body, fontName=FONTS["bold"], fontSize=12.5,
            leading=15, spaceBefore=13, spaceAfter=5, alignment=TA_LEFT,
        ),
        "h2": ParagraphStyle(
            "H2", parent=body, fontName=FONTS["bold"], fontSize=10.8,
            leading=13, spaceBefore=9, spaceAfter=4, alignment=TA_LEFT,
        ),
        "h3": ParagraphStyle(
            "H3", parent=body, fontName=FONTS["bi"], fontSize=10.2,
            leading=12.6, spaceBefore=7, spaceAfter=3, alignment=TA_LEFT,
        ),
        "body": body,
        "bullet": ParagraphStyle(
            "Bullet", parent=body, leftIndent=14, bulletIndent=4,
            spaceAfter=3, alignment=TA_JUSTIFY,
        ),
        "abstract": ParagraphStyle(
            "Abstract", parent=body, fontSize=9.6, leading=12.8,
            leftIndent=0.7 * cm, rightIndent=0.7 * cm, spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=body, fontName=FONTS["body"], fontSize=8.6,
            leading=11, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10,
        ),
        "cell": ParagraphStyle(
            "Cell", parent=body, fontName=FONTS["body"], fontSize=8.2,
            leading=10, alignment=TA_CENTER, spaceAfter=0,
        ),
        "cellL": ParagraphStyle(
            "CellL", parent=body, fontName=FONTS["body"], fontSize=8.2,
            leading=10, alignment=TA_LEFT, spaceAfter=0,
        ),
        "ref": ParagraphStyle(
            "Ref", parent=body, fontSize=8.8, leading=11.6,
            leftIndent=16, firstLineIndent=-16, spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "Code", parent=body, fontName=FONTS["mono"], fontSize=8.4,
            leading=10.4, alignment=TA_LEFT, leftIndent=10,
            backColor=colors.HexColor("#f4f4f4"), borderPadding=4,
            spaceBefore=4, spaceAfter=8,
        ),
    }


# --------------------------------------------------------------------------- #
# inline markdown -> reportlab markup
# --------------------------------------------------------------------------- #
def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)(\{[^}]*\})?", "", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", rf'<font face="{FONTS["mono"]}" size="8.6">\1</font>', text)
    return text


# --------------------------------------------------------------------------- #
# block parser
# --------------------------------------------------------------------------- #
def parse_front_matter(lines: list[str]) -> tuple[dict, list[str]]:
    meta: dict[str, str] = {}
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return meta, lines[i + 1 :]
            if ":" in lines[i]:
                k, v = lines[i].split(":", 1)
                meta[k.strip().lower()] = v.strip()
    return meta, lines


def make_table(rows: list[list[str]], styles) -> Table:
    header, body = rows[0], rows[1:]
    n_cols = len(header)
    hdr_style = ParagraphStyle(
        "Hdr", parent=styles["cell"], fontName=FONTS["bold"], fontSize=8.2,
        leading=10,
    )
    data = [[Paragraph(inline(c), hdr_style) for c in header]]
    for r in body:
        r = (r + [""] * n_cols)[:n_cols]
        data.append([
            Paragraph(inline(c), styles["cellL"] if i == 0 else styles["cell"])
            for i, c in enumerate(r)
        ])

    t = Table(data, hAlign="CENTER", repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return t


def make_figure(path: str, caption: str, width_frac: float) -> list:
    p = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    if not p.exists():
        return [
            Paragraph(
                inline(f"[missing figure: {path}]"),
                ParagraphStyle("miss", fontName=FONTS["italic"], fontSize=9,
                               alignment=TA_CENTER, textColor=colors.grey),
            )
        ]
    iw, ih = ImageReader(str(p)).getSize()
    w = CONTENT_W * width_frac
    h = w * ih / iw
    max_h = PAGE_H - M_TOP - M_BOTTOM - 3 * cm
    if h > max_h:
        h = max_h
        w = h * iw / ih
    return [
        Spacer(1, 4),
        KeepTogether([
            Image(str(p), width=w, height=h),
            Paragraph(inline(caption), build_styles()["caption"]),
        ]),
    ]


def build_story(md: str, styles) -> tuple[dict, list]:
    lines = md.splitlines()
    meta, lines = parse_front_matter(lines)

    story: list = []
    i = 0
    n = len(lines)
    guard = 0

    if meta.get("title"):
        story.append(Paragraph(inline(meta["title"]), styles["title"]))
    if meta.get("authors"):
        story.append(Paragraph(inline(meta["authors"]), styles["authors"]))
    if meta.get("affiliation"):
        story.append(Paragraph(inline(meta["affiliation"]), styles["affil"]))

    while i < n:
        guard += 1
        if guard > n * 8 + 100:
            raise RuntimeError(f"parser stuck at line {i}: {lines[i][:80]!r}")
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # html comments (used as result-injection markers) are not rendered
        if stripped.startswith("<!--"):
            while i < n and "-->" not in lines[i]:
                i += 1
            i += 1
            continue

        if stripped == "---page---":
            story.append(PageBreak())
            i += 1
            continue

        # fenced code
        if stripped.startswith("```"):
            j = i + 1
            buf = []
            while j < n and not lines[j].strip().startswith("```"):
                buf.append(lines[j])
                j += 1
            code = "<br/>".join(
                l.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&#160;")
                for l in buf
            )
            story.append(Paragraph(code, styles["code"]))
            i = j + 1
            continue

        # headings
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            if level == 1:
                story.append(Paragraph(inline(text), styles["h1"]))
            elif level == 2:
                story.append(Paragraph(inline(text), styles["h2"]))
            else:
                story.append(Paragraph(inline(text), styles["h3"]))
            i += 1
            continue

        # abstract marker
        if stripped.lower() in {"**abstract**", "abstract"}:
            j = i + 1
            while j < n and not lines[j].strip():  # skip the blank line after the label
                j += 1
            buf = []
            while (
                j < n
                and lines[j].strip()
                and not lines[j].strip().startswith("#")
                and lines[j].strip() != "---page---"
            ):
                buf.append(lines[j].strip())
                j += 1
            story.append(Paragraph(
                "<b>Abstract</b>&#8212; " + inline(" ".join(buf)), styles["abstract"]))
            i = j
            continue

        # table
        if stripped.startswith("|") and i + 1 < n and re.match(
            r"^\|[\s:|-]+\|$", lines[i + 1].strip()
        ):
            rows = []
            j = i
            while j < n and lines[j].strip().startswith("|"):
                cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                if not re.match(r"^[\s:|-]+$", "-".join(cells)):
                    rows.append(cells)
                j += 1
            story.append(Spacer(1, 4))
            story.append(make_table(rows, styles))
            story.append(Spacer(1, 8))
            i = j
            continue

        # figure
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)(?:\{([^}]*)\})?", stripped)
        if m:
            caption = m.group(1)
            path = m.group(2)
            opts = m.group(3) or ""
            wm = re.search(r"w=([\d.]+)", opts)
            width_frac = float(wm.group(1)) if wm else 0.9
            story.extend(make_figure(path, caption, width_frac))
            i += 1
            continue

        # bullets
        if re.match(r"^[-*]\s+", stripped):
            j = i
            while j < n and re.match(r"^[-*]\s+", lines[j].strip()):
                text = re.sub(r"^[-*]\s+", "", lines[j].strip())
                story.append(Paragraph(inline(text), styles["bullet"], bulletText="•"))
                j += 1
            story.append(Spacer(1, 3))
            i = j
            continue

        # numbered
        if re.match(r"^\d+\.\s+", stripped):
            j = i
            num = 1
            while j < n and re.match(r"^\d+\.\s+", lines[j].strip()):
                text = re.sub(r"^\d+\.\s+", "", lines[j].strip())
                story.append(Paragraph(inline(text), styles["bullet"], bulletText=f"{num}."))
                j += 1
                num += 1
            story.append(Spacer(1, 3))
            i = j
            continue

        # reference item  [1] ...
        if re.match(r"^\[\d+\]", stripped):
            j = i + 1
            while j < n and lines[j].strip() and not re.match(
                r"^\[\d+\]", lines[j].strip()
            ):
                j += 1
            para = " ".join(l.strip() for l in lines[i:j])
            story.append(Paragraph(inline(para), styles["ref"]))
            i = j
            continue

        # paragraph
        j = i
        buf = []
        while j < n and lines[j].strip() and not re.match(
            r"^(#{1,4}\s|[-*]\s|\d+\.\s|\||```|!\[|---page---|\[\d+\])", lines[j].strip()
        ):
            buf.append(lines[j].strip())
            j += 1
        if not buf:
            buf = [stripped]
            j = i + 1
        story.append(Paragraph(inline(" ".join(buf)), styles["body"]))
        i = j

    return meta, story


# --------------------------------------------------------------------------- #
# document
# --------------------------------------------------------------------------- #
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONTS["body"], 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawCentredString(PAGE_W / 2, 1.15 * cm, str(doc.page))
    canvas.restoreState()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(ROOT / "paper" / "paper.md"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "CoolKarachi.pdf"))
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"missing {src}")
    md = src.read_text(encoding="utf-8")
    register_fonts()
    styles = build_styles()
    meta, story = build_story(md, styles)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = BaseDocTemplate(
        str(out),
        pagesize=A4,
        title=meta.get("title", "CoolKarachi"),
        author=meta.get("authors", ""),
        leftMargin=M_LEFT,
        rightMargin=M_RIGHT,
        topMargin=M_TOP,
        bottomMargin=M_BOTTOM,
    )
    frame = Frame(
        M_LEFT, M_BOTTOM, CONTENT_W, PAGE_H - M_TOP - M_BOTTOM,
        id="body", showBoundary=0,
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=on_page)])
    doc.build(story)
    pages = doc.page
    print(f"PDF written -> {out}  ({pages} pages)")


if __name__ == "__main__":
    main()
