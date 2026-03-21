import io
import streamlit as st
from datetime import datetime

from src.comments.extract import extract_comments, extract_paragraphs
from src.comments.render import render_comments
from src.redlines.extract import extract_redlines, extract_moves
from src.stats.compute import comment_summary, redline_summary
from src.stats.render import render_comment_summary, render_redline_summary


# ---------------------------------------------------------------------------
# Session state — initialise permanent keys once
# ---------------------------------------------------------------------------
DEFAULTS: dict = {
    "p1_is_closed":   False,
    "p1_closed_date": None,
    "p1_order":       ["Sentence", "Comment"],
    "p1_file_bytes":  None,
    "p1_file_name":   None,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def _store_uploaded_file():
    f = st.session_state["_p1_uploaded_file"]
    if f is not None:
        st.session_state["p1_file_bytes"] = f.read()
        st.session_state["p1_file_name"]  = f.name
        f.seek(0)
    else:
        st.session_state["p1_file_bytes"] = None
        st.session_state["p1_file_name"]  = None

def _store_is_closed():   st.session_state["p1_is_closed"]   = st.session_state["_p1_is_closed"]
def _store_closed_date(): st.session_state["p1_closed_date"]  = st.session_state["_p1_closed_date"]
def _store_order():       st.session_state["p1_order"]        = st.session_state["_p1_order"]


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


def _load_document(file_bytes: bytes) -> tuple:
    """
    Extract all data from file bytes, creating a fresh BytesIO for each call.

    Returns
    -------
    comments       : list[Comment]
    version        : WordVersion
    redlines       : list[Redline]
    moves          : list[Move]
    doc_paragraphs : DocumentParagraphs
                     .paragraphs — final document paragraph texts (moveFrom excluded)
                     .moved_from — xml_order_idx → text for moved-away paragraphs
    """
    comments, version = extract_comments(io.BytesIO(file_bytes))
    redlines, _       = extract_redlines(io.BytesIO(file_bytes))
    moves, _          = extract_moves(io.BytesIO(file_bytes))
    doc_paragraphs    = extract_paragraphs(io.BytesIO(file_bytes))
    return comments, version, redlines, moves, doc_paragraphs


def _sidebar_controls(comments, redlines) -> tuple[list[str], datetime | None]:
    """Render sidebar controls. Returns (element_order, reference_date)."""
    st.sidebar.markdown("### Document")

    # --- Matter closed toggle ---
    st.session_state["_p1_is_closed"] = st.session_state["p1_is_closed"]
    is_closed = st.sidebar.toggle(
        "Matter is closed",
        key="_p1_is_closed",
        on_change=_store_is_closed,
    )

    # --- Closed date input ---
    reference_date = None
    if is_closed:
        latest  = _latest_date(comments, redlines)
        default = latest.date() if latest else datetime.today().date()
        saved   = st.session_state["p1_closed_date"]
        st.session_state["_p1_closed_date"] = saved if saved is not None else default
        closed_date = st.sidebar.date_input(
            "Closed date",
            key="_p1_closed_date",
            on_change=_store_closed_date,
        )
        reference_date = datetime.combine(closed_date, datetime.min.time())

    # --- Comment elements multiselect ---
    st.sidebar.markdown("### Comments")
    ALL_ELEMENTS = ["Sentence", "Comment", "Paragraph"]
    st.session_state["_p1_order"] = st.session_state["p1_order"]
    order = st.sidebar.multiselect(
        "Elements to show",
        options=ALL_ELEMENTS,
        key="_p1_order",
        on_change=_store_order,
    )

    return order, reference_date


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.sidebar.file_uploader(
    "Choose a file",
    type=["docx", "doc"],
    key="_p1_uploaded_file",
    on_change=_store_uploaded_file,
)

file_bytes = st.session_state["p1_file_bytes"]

# selection = st.pills("Show", ["Overview", "Comments"], selection_mode="single")

if file_bytes:
    comments, version, redlines, moves, doc_paragraphs = _load_document(file_bytes)
    order, reference_date = _sidebar_controls(comments, redlines)

    # match selection:
    #     case "Overview":
    st.markdown("## Comments")
    render_comment_summary(
        comment_summary(comments, reference_date=reference_date)
    )
    st.markdown("## Redlines")
    render_redline_summary(
        redline_summary(redlines, doc_paragraphs.paragraphs, reference_date=reference_date)
    )

        # case "Comments":
        #     render_comments(comments, version, order=order)
