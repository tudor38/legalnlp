"""
Typed session-state accessors for the legalnlp app.

All session-state key strings live here as module-level constants.
Cross-page keys (the ones read or written by more than one file) also get
typed get/set helpers so callers never deal with raw string keys.

Page-local keys (filter state, timeline view state, etc.) are defined as
constants so a rename is a single change here rather than a grep-and-replace.

Usage
-----
    # Reading the uploaded document (cross-page):
    from src.app_state import get_file_bytes
    file_bytes = get_file_bytes()   # bytes | None

    # Writing (document_statistics on upload):
    from src.app_state import set_file_bytes, set_file_name
    set_file_bytes(uploaded.read())
    set_file_name(uploaded.name)

    # Using a local key constant:
    from src.app_state import KEY_DOC_FINALIZED
    st.session_state[KEY_DOC_FINALIZED] = True
"""

from __future__ import annotations

from datetime import date

import streamlit as st

# ---------------------------------------------------------------------------
# Key constants — the ONLY place these strings should appear
# ---------------------------------------------------------------------------

# Cross-page: set by document_statistics, read by all analysis pages
KEY_DOC_BYTES = "doc_bytes"
KEY_DOC_NAME = "doc_name"

# Document status
KEY_DOC_FINALIZED = "doc_finalized"
KEY_DOC_FINALIZED_DATE = "doc_finalized_date"

# Sidebar filters
KEY_FILTER_DATE_MIN = "filter_date_min"
KEY_FILTER_DATE_MAX = "filter_date_max"
KEY_FILTER_AUTHORS = "filter_authors"

# Statistics page — tab selection
KEY_STATS_MAIN_TAB = "stats_main_tab"
KEY_COMMENT_VIEW = "comment_view"
KEY_REDLINE_VIEW = "redline_view"

# Statistics page — comment timeline view state
KEY_COMMENT_TL_EXPANDED = "comment_tl_expanded"
KEY_COMMENT_TL_EXPAND_ALL = "comment_tl_expand_all"
KEY_COMMENT_TL_FIELDS = "comment_tl_fields"

# Statistics page — redline timeline view state
KEY_REDLINE_TL_EXPANDED = "redline_tl_expanded"
KEY_REDLINE_TL_EXPAND_ALL = "redline_tl_expand_all"
KEY_REDLINE_TL_FIELDS = "redline_tl_fields"

# Statistics page — move view/timeline state
KEY_MOVE_VIEW = "move_view"
KEY_MOVE_TL_EXPANDED = "move_tl_expanded"
KEY_MOVE_TL_EXPAND_ALL = "move_tl_expand_all"
KEY_MOVE_TL_FIELDS = "move_tl_fields"


# ---------------------------------------------------------------------------
# Model name constants
# ---------------------------------------------------------------------------

# Sentence transformer models (used in search, topic explorer)
MODEL_MINILM = "all-MiniLM-L6-v2"
MODEL_MPNET = "all-mpnet-base-v2"
SENTENCE_TRANSFORMER_MODELS = [MODEL_MINILM, MODEL_MPNET]


# ---------------------------------------------------------------------------
# Cross-page typed accessors
# ---------------------------------------------------------------------------


def get_file_bytes() -> bytes | None:
    return st.session_state.get(KEY_DOC_BYTES)


def set_file_bytes(value: bytes | None) -> None:
    st.session_state[KEY_DOC_BYTES] = value


def get_file_name() -> str | None:
    return st.session_state.get(KEY_DOC_NAME)


def set_file_name(value: str | None) -> None:
    st.session_state[KEY_DOC_NAME] = value


def get_date_range() -> tuple[date | None, date | None]:
    return (
        st.session_state.get(KEY_FILTER_DATE_MIN),
        st.session_state.get(KEY_FILTER_DATE_MAX),
    )
