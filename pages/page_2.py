import io
import streamlit as st
from datetime import datetime, timedelta

from src.comments.extract import extract_comments, extract_paragraphs
from src.redlines.extract import extract_redlines, extract_moves
from src.stats.compute import (
    comment_ages_df,
    redline_ages_df,
    filter_by_date,
    latest_date,
    problematic_passages,
)
from src.stats.render import render_problematic_passages
from src.stats.config import CFG

_CLOSED_DATE_OFFSET = CFG["display"]["closed_date_offset_days"]
_TOP_N_OPTIONS = [5, 10, 20, 50, "All"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_document(file_bytes: bytes) -> tuple:
    comments, version = extract_comments(io.BytesIO(file_bytes))
    redlines, _ = extract_redlines(io.BytesIO(file_bytes))
    moves, _ = extract_moves(io.BytesIO(file_bytes))
    doc_paragraphs = extract_paragraphs(io.BytesIO(file_bytes))
    return comments, version, redlines, moves, doc_paragraphs


def _reference_date() -> tuple[datetime, bool]:
    """Reconstruct reference_date and is_closed from page 1 session state."""
    is_closed = st.session_state.get("p1_is_closed", False)
    if is_closed:
        closed_date = st.session_state.get("p1_closed_date")
        if closed_date:
            return (
                datetime.combine(closed_date, datetime.min.time())
                + timedelta(days=_CLOSED_DATE_OFFSET),
                True,
            )
    return datetime.now(), False


def _date_range(c_df, r_df) -> tuple:
    """Reconstruct date range from page 1 session state."""
    non_empty = [df for df in [c_df, r_df] if not df.empty]
    if not non_empty:
        today = datetime.now().date()
        return today, today

    global_min = min(df["date"].min().date() for df in non_empty)
    global_max = datetime.now().date()

    saved_min = st.session_state.get("p1_date_min") or global_min
    saved_max = st.session_state.get("p1_date_max") or global_max
    return max(saved_min, global_min), min(saved_max, global_max)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
file_bytes = st.session_state.get("p1_file_bytes")

if not file_bytes:
    st.info("Upload a document on Page 1 to get started.")
    st.stop()

comments, version, redlines, moves, doc_paragraphs = _load_document(file_bytes)
reference_date, is_closed = _reference_date()

c_df = comment_ages_df(comments, reference_date)
r_df = redline_ages_df(redlines, reference_date)

date_min, date_max = _date_range(c_df, r_df)

# Apply date filter to get activity within the selected range
filtered_c_df = filter_by_date(c_df, date_min, date_max)
filtered_r_df = filter_by_date(r_df, date_min, date_max)

# Derive filtered comment/redline lists by para_idx
active_comment_paras = (
    set(filtered_c_df["para_idx"].dropna().astype(int).tolist())
    if "para_idx" in filtered_c_df.columns
    else None
)
active_redline_paras = (
    set(filtered_r_df["para_idx"].dropna().astype(int).tolist())
    if "para_idx" in filtered_r_df.columns
    else None
)

# Compute all passages then filter by date-active paragraphs
all_passages = problematic_passages(
    comments=comments,
    redlines=redlines,
    moves=moves,
    all_paragraphs=doc_paragraphs.paragraphs,
)

# Date caption
if not c_df.empty or not r_df.empty:
    non_empty = [df for df in [c_df, r_df] if not df.empty]
    earliest = min(df["date"].min() for df in non_empty).strftime(
        CFG["display"]["date_format"]
    )
    end = (
        max(df["date"].max() for df in non_empty).strftime(
            CFG["display"]["date_format"]
        )
        if is_closed
        else reference_date.strftime(CFG["display"]["date_format"])
    )
    st.caption(f"From {earliest} → {end}")

# Top N selector
top_n_label = st.selectbox(
    "Show top",
    options=_TOP_N_OPTIONS,
    index=1,  # default: 10
    key="p2_top_n",
)

passages = all_passages if top_n_label == "All" else all_passages[: int(top_n_label)]

render_problematic_passages(passages)
