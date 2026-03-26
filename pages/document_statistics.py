import io
from typing import NamedTuple
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd

from src.comments.extract import extract_comments, extract_paragraphs
from src.redlines.extract import extract_redlines, extract_moves
from src.shared import DocxParseError
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
    render_timeline,
    render_date_caption,
    COMMENT_FIELDS,
    REDLINE_FIELDS,
    MOVE_FIELDS,
)
from src.app_state import (
    KEY_DOC_BYTES,
    KEY_DOC_FINALIZED,
    KEY_DOC_FINALIZED_DATE,
    KEY_DOC_NAME,
    KEY_COMMENT_TL_EXPAND_ALL,
    KEY_COMMENT_TL_EXPANDED,
    KEY_COMMENT_TL_FIELDS,
    KEY_COMMENT_VIEW,
    KEY_FILTER_AUTHORS,
    KEY_FILTER_DATE_MAX,
    KEY_FILTER_DATE_MIN,
    KEY_REDLINE_TL_EXPAND_ALL,
    KEY_REDLINE_TL_EXPANDED,
    KEY_REDLINE_TL_FIELDS,
    KEY_REDLINE_VIEW,
    KEY_MOVE_TL_EXPAND_ALL,
    KEY_MOVE_TL_EXPANDED,
    KEY_MOVE_TL_FIELDS,
    KEY_MOVE_VIEW,
    KEY_STATS_MAIN_TAB,
    get_file_bytes,
    set_file_bytes,
    set_file_name,
)
from src.stats.config import CFG

_ALLOWED_FILETYPES = CFG.display.allowed_filetypes
_CLOSED_DATE_OFFSET = CFG.display.closed_date_offset_days


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


class RedlineViews(NamedTuple):
    counts: str
    timeline: str


class MoveViews(NamedTuple):
    counts: str
    timeline: str


MAIN_TABS = MainTabs(*CFG.page_1_tabs.main)
COMMENT_VIEWS = CommentViews(*CFG.page_1_tabs.comment_views)
REDLINE_VIEWS = RedlineViews(*CFG.page_1_tabs.redline_views)
MOVE_VIEWS = MoveViews(*CFG.page_1_tabs.move_views)


# ---------------------------------------------------------------------------
# Session state — initialise permanent keys once
# ---------------------------------------------------------------------------
DEFAULTS: dict = {
    KEY_DOC_FINALIZED: False,
    KEY_DOC_FINALIZED_DATE: None,
    KEY_DOC_BYTES: None,
    KEY_DOC_NAME: None,
    # comment timeline
    KEY_COMMENT_TL_EXPANDED: False,
    KEY_COMMENT_TL_EXPAND_ALL: True,
    KEY_COMMENT_TL_FIELDS: [f.label for f in COMMENT_FIELDS],
    # redline timeline
    KEY_REDLINE_TL_EXPANDED: False,
    KEY_REDLINE_TL_EXPAND_ALL: True,
    KEY_REDLINE_TL_FIELDS: [f.label for f in REDLINE_FIELDS],
    # move timeline
    KEY_MOVE_TL_EXPANDED: False,
    KEY_MOVE_TL_EXPAND_ALL: True,
    KEY_MOVE_TL_FIELDS: [f.label for f in MOVE_FIELDS],
    # filters
    KEY_FILTER_AUTHORS: [],
    KEY_FILTER_DATE_MIN: None,
    KEY_FILTER_DATE_MAX: None,
    # tabs
    KEY_STATS_MAIN_TAB: MAIN_TABS.comments,
    KEY_COMMENT_VIEW: COMMENT_VIEWS.counts,
    KEY_REDLINE_VIEW: REDLINE_VIEWS.counts,
    KEY_MOVE_VIEW: MOVE_VIEWS.counts,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _make_store(perm_key: str, default=None):
    def callback():
        st.session_state[perm_key] = st.session_state.get(f"_{perm_key}", default)

    return callback


def _seed(perm_key: str) -> None:
    st.session_state[f"_{perm_key}"] = st.session_state[perm_key]


_store_is_closed = _make_store(KEY_DOC_FINALIZED, False)
_store_closed_date = _make_store(KEY_DOC_FINALIZED_DATE, None)
_store_expanded_view = _make_store(KEY_COMMENT_TL_EXPANDED, False)
_store_expand_all = _make_store(KEY_COMMENT_TL_EXPAND_ALL, False)
_store_show_fields = _make_store(KEY_COMMENT_TL_FIELDS, [])
_store_r_expanded_view = _make_store(KEY_REDLINE_TL_EXPANDED, False)
_store_r_expand_all = _make_store(KEY_REDLINE_TL_EXPAND_ALL, False)
_store_r_show_fields = _make_store(KEY_REDLINE_TL_FIELDS, [])
_store_m_expanded_view = _make_store(KEY_MOVE_TL_EXPANDED, False)
_store_m_expand_all = _make_store(KEY_MOVE_TL_EXPAND_ALL, False)
_store_m_show_fields = _make_store(KEY_MOVE_TL_FIELDS, [])
_store_timeline_authors = _make_store(KEY_FILTER_AUTHORS, [])
_store_main_tab = _make_store(KEY_STATS_MAIN_TAB, MAIN_TABS.comments)
_store_comment_tab = _make_store(KEY_COMMENT_VIEW, COMMENT_VIEWS.counts)
_store_redline_tab = _make_store(KEY_REDLINE_VIEW, REDLINE_VIEWS.counts)
_store_move_tab = _make_store(KEY_MOVE_VIEW, MOVE_VIEWS.counts)


def _store_date_range():
    val = st.session_state.get("_filter_date_range")
    if val:
        st.session_state[KEY_FILTER_DATE_MIN] = val[0]
        st.session_state[KEY_FILTER_DATE_MAX] = val[1]


def _store_uploaded_file():
    f = st.session_state.get("_doc_upload")
    if f is not None:
        st.cache_data.clear()
        set_file_bytes(f.read())
        set_file_name(f.name)
        for key, val in DEFAULTS.items():
            if key not in (KEY_DOC_BYTES, KEY_DOC_NAME):
                st.session_state[key] = val
        for widget_key in (
            "_filter_date_range",
            "_filter_authors",
            "_doc_finalized",
            "_doc_finalized_date",
            "_stats_main_tab",
            "_comment_view",
            "_redline_view",
            "_comment_tl_expanded",
            "_comment_tl_expand_all",
            "_comment_tl_fields",
            "_redline_tl_expanded",
            "_redline_tl_expand_all",
            "_redline_tl_fields",
            "_move_view",
            "_move_tl_expanded",
            "_move_tl_expand_all",
            "_move_tl_fields",
        ):
            if widget_key in st.session_state:
                del st.session_state[widget_key]
        f.seek(0)
    else:
        set_file_bytes(None)
        set_file_name(None)


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=3)
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
    st.sidebar.markdown("### Document")
    _seed(KEY_DOC_FINALIZED)
    is_closed = st.sidebar.toggle(
        "Matter is closed",
        key="_doc_finalized",
        on_change=_store_is_closed,
        help="Toggle ON if the document is part of a closed matter. This setting affects date calculations.",
    )

    if is_closed:
        ld = latest_date(comments, redlines)
        default = ld.date() if ld else datetime.today().date()
        saved = st.session_state[KEY_DOC_FINALIZED_DATE]
        st.session_state["_doc_finalized_date"] = saved if saved is not None else default
        closed_date = st.sidebar.date_input(
            "Closed date",
            key="_doc_finalized_date",
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

    saved_min = max(st.session_state[KEY_FILTER_DATE_MIN] or global_date_min, global_date_min)
    saved_max = min(st.session_state[KEY_FILTER_DATE_MAX] or global_date_max, global_date_max)

    date_range = st.sidebar.slider(
        "Date range",
        min_value=global_date_min,
        max_value=global_date_max,
        value=(saved_min, saved_max),
        key="_filter_date_range",
        on_change=_store_date_range,
    )

    c_df = all_dfs[0]
    if not c_df.empty:
        all_authors = sorted(c_df["author"].unique().tolist())
        st.session_state["_filter_authors"] = (
            st.session_state[KEY_FILTER_AUTHORS] or all_authors
        )
        selected_authors = (
            st.sidebar.multiselect(
                "Comment authors",
                options=all_authors,
                key="_filter_authors",
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
def _render_comment_timeline(
    filtered_c_df: pd.DataFrame,
    all_authors: list[str],
) -> None:
    for key in (KEY_COMMENT_TL_EXPANDED, KEY_COMMENT_TL_EXPAND_ALL, KEY_COMMENT_TL_FIELDS):
        _seed(key)
    render_timeline(
        filtered_c_df,
        "Who commented? When?",
        fields=COMMENT_FIELDS,
        display_cols=[
            "author",
            "date",
            "kind",
            "resolved",
            "comment",
            "selected",
            "sentence",
            "paragraph",
        ],
        default_fields=["Resolved", "Sentence", "Comment"],
        all_authors=all_authors,
        expanded_view_key="_comment_tl_expanded",
        expand_all_key="_comment_tl_expand_all",
        show_fields_key="_comment_tl_fields",
        on_expanded_view=_store_expanded_view,
        on_expand_all=_store_expand_all,
        on_show_fields=_store_show_fields,
    )


def _render_redline_timeline(
    filtered_r_df: pd.DataFrame,
    all_authors: list[str],
) -> None:
    for key in (KEY_REDLINE_TL_EXPANDED, KEY_REDLINE_TL_EXPAND_ALL, KEY_REDLINE_TL_FIELDS):
        _seed(key)
    render_timeline(
        filtered_r_df,
        "Who redlined? When?",
        fields=REDLINE_FIELDS,
        display_cols=["author", "date", "kind", "text", "sentence", "paragraph"],
        default_fields=["Redline", "Sentence"],
        all_authors=all_authors,
        expanded_view_key="_redline_tl_expanded",
        expand_all_key="_redline_tl_expand_all",
        show_fields_key="_redline_tl_fields",
        on_expanded_view=_store_r_expanded_view,
        on_expand_all=_store_r_expand_all,
        on_show_fields=_store_r_show_fields,
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

    _seed(KEY_COMMENT_VIEW)
    comment_tab = st.pills(
        "View",
        list(COMMENT_VIEWS),
        key="_comment_view",
        on_change=_store_comment_tab,
        selection_mode="single",
        label_visibility="collapsed",
    )

    comment_view_renderers = {
        COMMENT_VIEWS.counts: lambda: render_author_bar(
            filtered_c_df, "Who commented? How much?", all_authors=all_authors
        ),
        COMMENT_VIEWS.timeline: lambda: _render_comment_timeline(
            filtered_c_df, all_authors
        ),
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

    _seed(KEY_REDLINE_VIEW)
    redline_tab = st.pills(
        "View",
        list(REDLINE_VIEWS),
        key="_redline_view",
        on_change=_store_redline_tab,
        selection_mode="single",
        label_visibility="collapsed",
    )

    redline_view_renderers = {
        REDLINE_VIEWS.counts: lambda: render_author_bar(
            filtered_r_df, "Who redlined? How much?", all_authors=all_authors
        ),
        REDLINE_VIEWS.timeline: lambda: _render_redline_timeline(
            filtered_r_df, all_authors
        ),
    }
    if redline_tab in redline_view_renderers:
        redline_view_renderers[redline_tab]()


def _render_move_timeline(
    filtered_m_df: pd.DataFrame,
    all_authors: list[str],
) -> None:
    for key in (KEY_MOVE_TL_EXPANDED, KEY_MOVE_TL_EXPAND_ALL, KEY_MOVE_TL_FIELDS):
        _seed(key)
    render_timeline(
        filtered_m_df,
        "Who moved text? When?",
        fields=MOVE_FIELDS,
        display_cols=["author", "date", "text", "distance", "from_para_idx", "to_para_idx"],
        default_fields=[f.label for f in MOVE_FIELDS],
        all_authors=all_authors,
        expanded_view_key="_move_tl_expanded",
        expand_all_key="_move_tl_expand_all",
        show_fields_key="_move_tl_fields",
        on_expanded_view=_store_m_expanded_view,
        on_expand_all=_store_m_expand_all,
        on_show_fields=_store_m_show_fields,
    )


def _render_moves(
    filtered_m_df: pd.DataFrame,
    reference_date: datetime,
    is_closed: bool,
    all_authors: list[str],
) -> None:
    render_date_caption(filtered_m_df, reference_date, is_closed)
    st.metric("Total Moves", len(filtered_m_df), border=True)

    _seed(KEY_MOVE_VIEW)
    move_tab = st.pills(
        "View",
        list(MOVE_VIEWS),
        key="_move_view",
        on_change=_store_move_tab,
        selection_mode="single",
        label_visibility="collapsed",
    )

    move_view_renderers = {
        MOVE_VIEWS.counts: lambda: render_author_bar(
            filtered_m_df, "Who moved text? How much?", all_authors=all_authors
        ),
        MOVE_VIEWS.timeline: lambda: _render_move_timeline(
            filtered_m_df, all_authors
        ),
    }
    if move_tab in move_view_renderers:
        move_view_renderers[move_tab]()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
_stored_name = st.session_state.get(KEY_DOC_NAME)
if _stored_name and not st.session_state.get("_doc_upload"):
    st.sidebar.caption(f"📄 {_stored_name}")
    if st.sidebar.button("Clear file", key="stats_clear_file", use_container_width=True):
        set_file_bytes(None)
        set_file_name(None)
        st.rerun()
else:
    st.sidebar.file_uploader(
        "Choose a file",
        type=_ALLOWED_FILETYPES,
        key="_doc_upload",
        on_change=_store_uploaded_file,
    )

file_bytes = get_file_bytes()

if not file_bytes:
    st.info("Upload a Word document using the sidebar to get started.")
    st.stop()

if file_bytes:
    try:
        comments, version, redlines, moves, doc_paragraphs = _load_document(file_bytes)
    except DocxParseError as e:
        st.error(f"Could not read the uploaded file: {e}")
        st.stop()

    c_df = comment_ages_df(comments, datetime.now())
    r_df = redline_ages_df(redlines, datetime.now())
    m_df = move_ages_df(moves, datetime.now())

    reference_date, is_closed, date_range, selected_authors = _sidebar_controls(
        comments, redlines, [c_df, r_df, m_df]
    )

    c_df = comment_ages_df(comments, reference_date)
    r_df = redline_ages_df(redlines, reference_date)
    m_df = move_ages_df(moves, reference_date)

    all_authors = sorted(c_df["author"].unique().tolist()) if not c_df.empty else []

    filtered_c_df = filter_by_date(c_df, date_range[0], date_range[1])
    filtered_r_df = filter_by_date(r_df, date_range[0], date_range[1])
    filtered_m_df = filter_by_date(m_df, date_range[0], date_range[1])
    if selected_authors:
        if not filtered_c_df.empty:
            filtered_c_df = filtered_c_df[
                filtered_c_df["author"].isin(selected_authors)
            ].reset_index(drop=True)
        if not filtered_r_df.empty:
            filtered_r_df = filtered_r_df[
                filtered_r_df["author"].isin(selected_authors)
            ].reset_index(drop=True)
        if not filtered_m_df.empty:
            filtered_m_df = filtered_m_df[
                filtered_m_df["author"].isin(selected_authors)
            ].reset_index(drop=True)

    _seed(KEY_STATS_MAIN_TAB)
    main_tab = st.pills(
        "Section",
        list(MAIN_TABS),
        key="_stats_main_tab",
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
