import streamlit as st
from datetime import datetime

from src.comments.extract import extract_comments, extract_paragraphs
from src.comments.render import render_comments
from src.redlines.extract import extract_redlines
from src.stats.compute import comment_summary, redline_summary
from src.stats.render import render_comment_summary, render_redline_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _latest_date(comments, redlines) -> datetime | None:
    """Return the latest date across all comments and redlines."""
    dates = []
    for c in comments:
        try:
            dates.append(datetime.fromisoformat(c.date.rstrip("Z")))
        except ValueError:
            pass
    for r in redlines:
        try:
            dates.append(datetime.fromisoformat(r.date.rstrip("Z")))
        except ValueError:
            pass
    return max(dates) if dates else None


def _load_document(uploaded_file) -> tuple:
    """Extract all data from the uploaded file, seeking between reads."""
    comments, version = extract_comments(uploaded_file)

    uploaded_file.seek(0)
    redlines, _ = extract_redlines(uploaded_file)

    uploaded_file.seek(0)
    all_paragraphs = extract_paragraphs(uploaded_file)

    return comments, version, redlines, all_paragraphs


def _sidebar_controls(comments, redlines) -> tuple[list[str], datetime | None]:
    """Render sidebar controls. Returns (element_order, reference_date)."""
    st.sidebar.markdown("### Document")

    is_closed = st.sidebar.toggle("Matter is closed")
    reference_date = None
    if is_closed:
        latest = _latest_date(comments, redlines)
        default = latest.date() if latest else datetime.today().date()
        closed_date = st.sidebar.date_input("Closed date", value=default)
        reference_date = datetime.combine(closed_date, datetime.min.time())

    st.sidebar.markdown("### Comments")
    ALL_ELEMENTS = ["Sentence", "Comment", "Paragraph"]
    order = st.sidebar.multiselect(
        "Elements to show", options=ALL_ELEMENTS, default=ALL_ELEMENTS[:-1]
    )

    return order, reference_date


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["docx", "doc"])

# selection = st.pills("Show", ["Overview", "Comments"], selection_mode="single")

if uploaded_file:
    comments, version, redlines, all_paragraphs = _load_document(uploaded_file)
    order, reference_date = _sidebar_controls(comments, redlines)

    # match selection:
    #     case "Overview":
    st.markdown("## Comments")
    render_comment_summary(
        comment_summary(comments, reference_date=reference_date)
    )
    st.markdown("## Redlines")
    render_redline_summary(
        redline_summary(redlines, all_paragraphs, reference_date=reference_date)
    )

        # case "Comments":
        #     render_comments(comments, version, order=order)
