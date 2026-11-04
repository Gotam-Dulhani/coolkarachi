"""Deliverables: paper, slides, reports and the figure gallery."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import ui_common as c


def _download(path: Path, label: str, mime: str, key: str) -> None:
    data = c.file_bytes(path)
    if data:
        st.download_button(label, data, file_name=path.name, mime=mime, key=key)
    else:
        st.caption(f"`{path.name}` not found.")


@st.cache_data(show_spinner=False)
def _docx_text(path_str: str, mtime_ns: int) -> str:
    try:
        from docx import Document
    except Exception:
        return ""
    try:
        document = Document(path_str)
        return "\n".join(p.text for p in document.paragraphs if p.text.strip())
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def _pptx_titles(path_str: str, mtime_ns: int) -> list[str]:
    try:
        from pptx import Presentation
    except Exception:
        return []
    try:
        prs = Presentation(path_str)
    except Exception:
        return []
    titles = []
    for slide in prs.slides:
        title = ""
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                title = shape.text_frame.text.strip().splitlines()[0]
                break
        titles.append(title or "(untitled slide)")
    return titles


def _paper() -> None:
    pdf = c.PAPER / "CoolKarachi.pdf"
    md = c.PAPER / "paper.md"
    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        _download(pdf, "Download paper (PDF)", "application/pdf", "dl_paper_pdf")
    with col2:
        _download(md, "Download source (Markdown)", "text/markdown", "dl_paper_md")

    text = c.read_paper_markdown()
    if not text:
        st.info("`paper/paper.md` not found.")
        return

    st.caption(
        f"{len(text):,} characters · rendered from `paper/paper.md`. "
        "Use the download buttons above for the typeset PDF."
    )
    with st.container(height=720, border=True):
        st.markdown(text)


def _slides() -> None:
    path = c.SLIDES / "CoolKarachi.pptx"
    if not path.exists():
        st.info("`slides/CoolKarachi.pptx` not found.")
        return
    _download(
        path,
        "Download slide deck (.pptx)",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "dl_slides",
    )
    titles = _pptx_titles(str(path), path.stat().st_mtime_ns)
    st.metric("Slides", len(titles))
    if titles:
        st.markdown("**Deck outline**")
        for i, title in enumerate(titles, start=1):
            st.markdown(f"{i}. {title}")
    else:
        st.caption("Install `python-pptx` (already in requirements.txt) to list slide titles.")


def _reports() -> None:
    docs = [
        ("SUBMISSION.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("PROJECT_OVERVIEW.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("PROJECT_PROPOSAL.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ]
    for name, mime in docs:
        path = c.DOCS / name
        if not path.exists():
            continue
        with st.expander(name.replace(".docx", "").replace("_", " ").title(), expanded=False):
            _download(path, f"Download {name}", mime, f"dl_{name}")
            text = _docx_text(str(path), path.stat().st_mtime_ns)
            if text:
                st.markdown(text[:12000] + ("\n\n…" if len(text) > 12000 else ""))
            else:
                st.caption("Install `python-docx` to preview the document text in-browser.")

    xlsx = c.ROOT / "Topic_Submission_CoolKarachi.xlsx"
    if xlsx.exists():
        with st.expander("Topic submission spreadsheet"):
            _download(
                xlsx,
                "Download topic submission (.xlsx)",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "dl_xlsx",
            )


def _figures() -> None:
    figures = sorted(c.FIGS.glob("*.png"))
    if not figures:
        st.info("No figures found.")
        return
    choice = st.selectbox("Figure", [p.name for p in figures], key="doc_figure")
    path = c.FIGS / choice
    st.image(str(path), use_container_width=True)
    _download(path, "Download PNG", "image/png", "dl_doc_figure")


def _dataset() -> None:
    st.markdown(
        "The twelve yearly Earth Engine exports and the cleaned 360,000-row table "
        "(~197 MB) live in a public Google Drive folder. They are intentionally not "
        "committed to GitHub."
    )
    st.link_button(
        "Open the dataset folder",
        "https://drive.google.com/drive/folders/1ZbUw7r6Sor7C47uXzFwsOGu8laFZ32Cg",
    )
    st.markdown("")
    st.markdown(
        "| File | Rows | Purpose |\n"
        "|---|---|---|\n"
        "| `karachi_ml_<year>.csv` | ~60,000 | Built-up stratified ML sample with LON/LAT |\n"
        "| `karachi_features_<year>.csv` | ~38,500 | All-cover sample for the urban–rural contrast |\n"
        "| `karachi_urban_clean.csv` | ~360,000 | Concatenated, cleaned built-up sample |\n"
        "| `karachi_lst_<year>.tif` | raster | 30 m LST for mapping |"
    )
    st.code(
        "# download the CSVs into data/, then:\n"
        "streamlit run app.py",
        language="powershell",
    )
    st.caption(
        "The dashboard reads whatever `karachi_*.csv` files are present in `data/`, "
        "or accepts CSVs uploaded directly in the Data explorer."
    )


def render() -> None:
    c.hero(
        "Deliverables and downloads",
        "Paper &",
        "Documents",
        "Read the full manuscript, download the deck and reports, and browse every "
        "generated figure without leaving the dashboard.",
    )
    tabs = st.tabs(["Paper", "Slides", "Reports", "Figures", "Dataset"])
    with tabs[0]:
        _paper()
    with tabs[1]:
        _slides()
    with tabs[2]:
        _reports()
    with tabs[3]:
        _figures()
    with tabs[4]:
        _dataset()
