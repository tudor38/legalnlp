import io
from typing import NamedTuple
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd

from src.comments.extract import extract_comments, extract_paragraphs
from src.redlines.extract import extract_redlines, extract_moves
from src.stats.compute import (
    comment_ages_df,
    redline_ages_df,
    move_ages_df,
    comment_metrics,
    filter_by_date,
    latest_date,
)
from src.stats.render import (
    render_author_bar,
    render_comment_metrics,
    render_comment_timeline,
    render_date_caption,
)
from src.stats.config import CFG

_ALLOWED_FILETYPES = CFG["display"]["allowed_filetypes"]
_CLOSED_DATE_OFFSET = CFG["display"]["closed_date_offset_days"]


# ---------------------------------------------------------------------------
# Tab structures
# ---------------------------------------------------------------------------
class MainTabs(NamedTuple):
    comments: str
    redlines: str
    moves: str


class CommentViews(NamedTuple):
    counts: str
    timeline: str


MAIN_TABS = MainTabs(*CFG["pages"]["page_1"]["tabs"]["main"])
COMMENT_VIEWS = CommentViews(*CFG["pages"]["page_1"]["tabs"]["comment_views"])


# ---------------------------------------------------------------------------
# Session state — initialise permanent keys once
# ---------------------------------------------------------------------------
DEFAULTS: dict = {
    "p1_is_closed": False,
    "p1_closed_date": None,
    "p1_file_bytes": None,
    "p1_file_name": None,
    "p1_expanded_view": False,
    "p1_expand_all": False,
    "p1_show_fields": ["Resolved", "Sentence", "Comment"],
    "p1_timeline_authors": [],
    "p1_date_min": None,
    "p1_date_max": None,
    "p1_main_tab": MAIN_TABS.comments,
    "p1_comment_tab": COMMENT_VIEWS.counts,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _make_store(perm_key: str, default=None):
    """Return a callback that copies the widget temp key into the permanent key."""

    def callback():
        st.session_state[perm_key] = st.session_state.get(f"_{perm_key}", default)

    return callback


def _seed(perm_key: str) -> None:
    """Copy permanent key → temp key before a widget renders."""
    st.session_state[f"_{perm_key}"] = st.session_state[perm_key]


_store_is_closed = _make_store("p1_is_closed", False)
_store_closed_date = _make_store("p1_closed_date", None)
_store_expanded_view = _make_store("p1_expanded_view", False)
_store_expand_all = _make_store("p1_expand_all", False)
_store_show_fields = _make_store("p1_show_fields", [])
_store_timeline_authors = _make_store("p1_timeline_authors", [])
_store_main_tab = _make_store("p1_main_tab", MAIN_TABS.comments)
_store_comment_tab = _make_store("p1_comment_tab", COMMENT_VIEWS.counts)


def _store_uploaded_file():
    f = st.session_state.get("_p1_uploaded_file")
    if f is not None:
        st.session_state["p1_file_bytes"] = f.read()
        st.session_state["p1_file_name"] = f.name
        f.seek(0)
    else:
        st.session_state["p1_file_bytes"] = None
        st.session_state["p1_file_name"] = None


def _store_date_range():
    val = st.session_state.get("_p1_date_range")
    if val:
        st.session_state["p1_date_min"] = val[0]
        st.session_state["p1_date_max"] = val[1]


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------
def _load_document(file_bytes: bytes) -> tuple:
    comments, version = extract_comments(io.BytesIO(file_bytes))
    redlines, _ = extract_redlines(io.BytesIO(file_bytes))
    moves, _ = extract_moves(io.BytesIO(file_bytes))
    doc_paragraphs = extract_paragraphs(io.BytesIO(file_bytes))
    return comments, version, redlines, moves, doc_paragraphs


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _sidebar_controls(
    comments, redlines, all_dfs: list[pd.DataFrame]
) -> tuple[datetime, bool, tuple, list[str]]:
    """
    Render sidebar controls.

    Returns
    -------
    reference_date   : datetime
    is_closed        : bool
    date_range       : tuple[date, date]
    selected_authors : list[str]
    """
    st.sidebar.markdown("### Document")
    _seed("p1_is_closed")
    is_closed = st.sidebar.toggle(
        "Matter is closed",
        key="_p1_is_closed",
        on_change=_store_is_closed,
        help="Toggle ON if the document is part of a closed matter. This setting affects date calculations.",
    )

    if is_closed:
        ld = latest_date(comments, redlines)
        default = ld.date() if ld else datetime.today().date()
        saved = st.session_state["p1_closed_date"]
        st.session_state["_p1_closed_date"] = saved if saved is not None else default
        closed_date = st.sidebar.date_input(
            "Closed date",
            key="_p1_closed_date",
            on_change=_store_closed_date,
        )
        reference_date = datetime.combine(closed_date, datetime.min.time()) + timedelta(
            days=_CLOSED_DATE_OFFSET
        )
    else:
        reference_date = datetime.now()

    non_empty = [df for df in all_dfs if not df.empty]
    if not non_empty:
        return (
            reference_date,
            is_closed,
            (datetime.now().date(), datetime.now().date()),
            [],
        )

    st.sidebar.markdown("### Filters")

    global_date_min = min(df["date"].min().date() for df in non_empty)
    global_date_max = (
        max(df["date"].max().date() for df in non_empty)
        if is_closed
        else datetime.now().date()
    )

    saved_min = max(st.session_state["p1_date_min"] or global_date_min, global_date_min)
    saved_max = min(st.session_state["p1_date_max"] or global_date_max, global_date_max)

    date_range = st.sidebar.slider(
        "Date range",
        min_value=global_date_min,
        max_value=global_date_max,
        value=(saved_min, saved_max),
        key="_p1_date_range",
        on_change=_store_date_range,
    )

    c_df = all_dfs[0]
    if not c_df.empty:
        all_authors = sorted(c_df["author"].unique().tolist())
        st.session_state["_p1_timeline_authors"] = (
            st.session_state["p1_timeline_authors"] or all_authors
        )
        selected_authors = (
            st.sidebar.multiselect(
                "Comment authors",
                options=all_authors,
                key="_p1_timeline_authors",
                on_change=_store_timeline_authors,
            )
            or all_authors
        )
    else:
        selected_authors = []

    return reference_date, is_closed, date_range, selected_authors


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------
def _render_timeline(
    filtered_c_df: pd.DataFrame,
    all_authors: list[str],
) -> None:
    for key in ("p1_expanded_view", "p1_expand_all", "p1_show_fields"):
        _seed(key)
    render_comment_timeline(
        filtered_c_df,
        "Who commented and when?",
        all_authors=all_authors,
        expanded_view_key="_p1_expanded_view",
        expand_all_key="_p1_expand_all",
        show_fields_key="_p1_show_fields",
        on_expanded_view=_store_expanded_view,
        on_expand_all=_store_expand_all,
        on_show_fields=_store_show_fields,
    )


def _render_comments(
    filtered_c_df: pd.DataFrame,
    reference_date: datetime,
    is_closed: bool,
    comments: list,
    all_authors: list[str],
) -> None:
    render_date_caption(filtered_c_df, reference_date, is_closed)
    render_comment_metrics(comment_metrics(comments))

    _seed("p1_comment_tab")
    comment_tab = st.pills(
        "View",
        list(COMMENT_VIEWS),
        key="_p1_comment_tab",
        on_change=_store_comment_tab,
        selection_mode="single",
        label_visibility="collapsed",
    )

    comment_view_renderers = {
        COMMENT_VIEWS.counts: lambda: render_author_bar(
            filtered_c_df, "Who commented and how much?", all_authors=all_authors
        ),
        COMMENT_VIEWS.timeline: lambda: _render_timeline(filtered_c_df, all_authors),
    }
    if comment_tab in comment_view_renderers:
        comment_view_renderers[comment_tab]()


def _render_redlines(
    filtered_r_df: pd.DataFrame,
    reference_date: datetime,
    is_closed: bool,
    all_authors: list[str],
) -> None:
    render_date_caption(filtered_r_df, reference_date, is_closed)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total", len(filtered_r_df), border=True)
    col2.metric(
        "Insertions",
        int((filtered_r_df["kind"] == "insertion").sum())
        if not filtered_r_df.empty
        else 0,
        border=True,
    )
    col3.metric(
        "Deletions",
        int((filtered_r_df["kind"] == "deletion").sum())
        if not filtered_r_df.empty
        else 0,
        border=True,
    )
    render_author_bar(filtered_r_df, "Count by Author", all_authors=all_authors)


def _render_moves(
    filtered_m_df: pd.DataFrame,
    reference_date: datetime,
    is_closed: bool,
    all_authors: list[str],
) -> None:
    render_date_caption(filtered_m_df, reference_date, is_closed)
    st.metric("Total Moves", len(filtered_m_df))
    render_author_bar(filtered_m_df, "Move Count by Author", all_authors=all_authors)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.sidebar.file_uploader(
    "Choose a file",
    type=_ALLOWED_FILETYPES,
    key="_p1_uploaded_file",
    on_change=_store_uploaded_file,
)

file_bytes = st.session_state["p1_file_bytes"]

if file_bytes:
    comments, version, redlines, moves, doc_paragraphs = _load_document(file_bytes)

    # Initial compute with now() to populate sidebar date slider
    c_df = comment_ages_df(comments, datetime.now())
    r_df = redline_ages_df(redlines, datetime.now())
    m_df = move_ages_df(moves, datetime.now())

    reference_date, is_closed, date_range, selected_authors = _sidebar_controls(
        comments, redlines, [c_df, r_df, m_df]
    )

    # Recompute with correct reference_date
    c_df = comment_ages_df(comments, reference_date)
    r_df = redline_ages_df(redlines, reference_date)
    m_df = move_ages_df(moves, reference_date)

    # Global author list — from unfiltered c_df for stable colors
    all_authors = sorted(c_df["author"].unique().tolist()) if not c_df.empty else []

    # Apply filters once
    filtered_c_df = filter_by_date(c_df, date_range[0], date_range[1])
    if not filtered_c_df.empty and selected_authors:
        filtered_c_df = filtered_c_df[
            filtered_c_df["author"].isin(selected_authors)
        ].reset_index(drop=True)
    filtered_r_df = filter_by_date(r_df, date_range[0], date_range[1])
    filtered_m_df = filter_by_date(m_df, date_range[0], date_range[1])

    # --- Main tab ---
    _seed("p1_main_tab")
    main_tab = st.pills(
        "Section",
        list(MAIN_TABS),
        key="_p1_main_tab",
        on_change=_store_main_tab,
        selection_mode="single",
        label_visibility="collapsed",
    )

    tab_renderers = {
        MAIN_TABS.comments: lambda: _render_comments(
            filtered_c_df, reference_date, is_closed, comments, all_authors
        ),
        MAIN_TABS.redlines: lambda: _render_redlines(
            filtered_r_df, reference_date, is_closed, all_authors
        ),
        MAIN_TABS.moves: lambda: _render_moves(
            filtered_m_df, reference_date, is_closed, all_authors
        ),
    }

    if main_tab in tab_renderers:
        st.divider()
        tab_renderers[main_tab]()
