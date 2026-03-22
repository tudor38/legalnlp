import io
import streamlit as st
from datetime import datetime, timedelta

from src.comments.extract import extract_comments, extract_paragraphs
from src.redlines.extract import extract_redlines, extract_moves
from src.stats.compute import (
    comment_ages_df,
    redline_ages_df,
    move_ages_df,
)
from src.stats.render import (
    render_author_bar,
    render_comment_metrics,
    render_comment_timeline,
)


# ---------------------------------------------------------------------------
# Session state — initialise permanent keys once
# ---------------------------------------------------------------------------
DEFAULTS: dict = {
    "p1_is_closed": False,
    "p1_closed_date": None,
    "p1_file_bytes": None,
    "p1_file_name": None,
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
        st.session_state["p1_file_name"] = f.name
        f.seek(0)
    else:
        st.session_state["p1_file_bytes"] = None
        st.session_state["p1_file_name"] = None


def _store_is_closed():
    st.session_state["p1_is_closed"] = st.session_state["_p1_is_closed"]


def _store_closed_date():
    st.session_state["p1_closed_date"] = st.session_state["_p1_closed_date"]


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
    """
    comments, version = extract_comments(io.BytesIO(file_bytes))
    redlines, _ = extract_redlines(io.BytesIO(file_bytes))
    moves, _ = extract_moves(io.BytesIO(file_bytes))
    doc_paragraphs = extract_paragraphs(io.BytesIO(file_bytes))
    return comments, version, redlines, moves, doc_paragraphs


def _sidebar_controls(comments, redlines) -> datetime:
    """Render sidebar controls. Returns reference_date."""

    st.session_state["_p1_is_closed"] = st.session_state["p1_is_closed"]
    is_closed = st.sidebar.toggle(
        "Matter is closed",
        key="_p1_is_closed",
        value=False,
        on_change=_store_is_closed,
        help="Toggle ON if the document is part of a closed matter. This setting affects date calculations.",
    )

    if is_closed:
        latest = _latest_date(comments, redlines)
        default = latest.date() if latest else datetime.today().date()
        saved = st.session_state["p1_closed_date"]
        st.session_state["_p1_closed_date"] = saved if saved is not None else default
        closed_date = st.sidebar.date_input(
            "Closed date",
            key="_p1_closed_date",
            on_change=_store_closed_date,
        )
        return datetime.combine(closed_date, datetime.min.time()) + timedelta(days=1)

    return datetime.now()


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

if file_bytes:
    comments, version, redlines, moves, doc_paragraphs = _load_document(file_bytes)
    reference_date = _sidebar_controls(comments, redlines)

    c_df = comment_ages_df(comments, reference_date)
    r_df = redline_ages_df(redlines, reference_date)
    m_df = move_ages_df(moves, reference_date)

    tab_c, tab_r, tab_m = st.tabs(["Comments", "Redlines", "Moves"])

    with tab_c:
        if not c_df.empty:
            earliest = c_df["date"].min().strftime("%B %-d, %Y")
            end = reference_date.strftime("%B %-d, %Y")
            st.caption(f"From {earliest} → {end}")

        total = len(comments)
        replies = sum(len(c.replies) for c in comments)
        resolved = sum(1 for c in comments if c.resolved) + sum(
            1 for c in comments for r in c.replies if r.resolved
        )

        render_comment_metrics(total + replies, resolved)
        render_author_bar(c_df, "Count by Author")
        render_comment_timeline(c_df, "Timeline")

    with tab_r:
        if not r_df.empty:
            earliest = r_df["date"].min().strftime("%B %-d, %Y")
            end = reference_date.strftime("%B %-d, %Y")
            st.caption(f"From {earliest} → {end}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(redlines))
        col2.metric("Insertions", sum(1 for r in redlines if r.kind == "insertion"))
        col3.metric("Deletions", sum(1 for r in redlines if r.kind == "deletion"))

        render_author_bar(r_df, "Count by Author")

    with tab_m:
        if not m_df.empty:
            earliest = m_df["date"].min().strftime("%B %-d, %Y")
            end = reference_date.strftime("%B %-d, %Y")
            st.caption(f"From {earliest} → {end}")

        st.metric("Total Moves", len(moves))

        render_author_bar(m_df, "Move Count by Author")
