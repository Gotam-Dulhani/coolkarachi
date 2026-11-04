"""CoolKarachi — Streamlit dashboard entrypoint.

Run from the project root:

    streamlit run app.py

The app is read-mostly: it renders the committed results, figures and documents,
and optionally loads the git-ignored CSV exports for interactive exploration and
what-if scenario simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# set_page_config must be the first Streamlit command.
st.set_page_config(
    page_title="CoolKarachi · Urban Heat Digital Twin",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "CoolKarachi — an explainable machine-learning digital twin for 30 m "
            "urban heat mapping and cooling-intervention prioritisation in Karachi."
        ),
        "Get help": "https://github.com/Gotam-Dulhani/coolkarachi",
    },
)

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ui_common as c  # noqa: E402
import ui_data_explorer  # noqa: E402
import ui_documents  # noqa: E402
import ui_maps  # noqa: E402
import ui_methodology  # noqa: E402
import ui_models  # noqa: E402
import ui_overview  # noqa: E402
import ui_scenarios  # noqa: E402

PAGES = {
    "Overview": ui_overview.render,
    "Data explorer": ui_data_explorer.render,
    "Model results": ui_models.render,
    "Cooling scenarios": ui_scenarios.render,
    "Heat maps": ui_maps.render,
    "Methodology": ui_methodology.render,
    "Paper & documents": ui_documents.render,
}


def _go_to_explorer() -> None:
    st.session_state["ck_nav"] = "Data explorer"


def _sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🌡️ CoolKarachi")
        st.caption("Explainable ML digital twin · 30 m surface heat · 2020–2025")
        st.divider()

        page = st.radio(
            "Navigate",
            list(PAGES),
            label_visibility="collapsed",
            key="ck_nav",
        )

        st.divider()
        st.markdown("**Session data**")
        df = c.get_dataset()
        if df is not None:
            st.success(f"{len(df):,} rows × {df.shape[1]} columns")
            source = c.get_dataset_source()
            if source:
                st.caption(source)
            if st.button("Clear loaded data", use_container_width=True):
                c.clear_dataset()
                st.rerun()
        else:
            st.caption("No CSV loaded. Data-dependent pages fall back to committed results.")
            st.button(
                "Go to Data explorer",
                use_container_width=True,
                on_click=_go_to_explorer,
            )

        st.divider()
        st.markdown("**Project**")
        st.link_button(
            "GitHub repository",
            "https://github.com/Gotam-Dulhani/coolkarachi",
            use_container_width=True,
        )
        st.link_button(
            "Dataset (Google Drive)",
            "https://drive.google.com/drive/folders/1ZbUw7r6Sor7C47uXzFwsOGu8laFZ32Cg",
            use_container_width=True,
        )
        st.caption("Gotam Dulhani · Mohat Kumar · FAST-NUCES Karachi")

    return page


def main() -> None:
    c.inject_css()
    page = _sidebar()
    PAGES[page]()

    st.divider()
    st.caption(
        "CoolKarachi · Course project, Department of Data Science, FAST-NUCES Karachi. "
        "Surface LST is not air temperature; scenarios are model-based counterfactuals, "
        "not field experiments."
    )


if __name__ == "__main__":
    main()
