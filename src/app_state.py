"""
Typed session-state accessors for the wordnlp app.

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
# Search page keys
# ---------------------------------------------------------------------------

KEY_SEARCH_STORED_FILES = "_search_stored_files"
KEY_SEARCH_PREF_PER_PAGE = "_search_pref_per_page"
KEY_SEARCH_PREF_QUERY = "_search_pref_query"
KEY_SEARCH_PREF_METHOD = "_search_pref_method"
KEY_SEARCH_PREF_MIN_SCORE = "_search_pref_min_score"
KEY_SEARCH_PREF_MODEL = "_search_pref_model"
KEY_SEARCH_PREF_CUSTOM_MODEL = "_search_pref_custom_model"
KEY_SEARCH_HITS = "_search_hits"
KEY_SEARCH_HITS_KEY = "_search_hits_key"
KEY_SEARCH_PAGE = "_search_page"
KEY_SEARCH_VIEW = "_search_view"

# ---------------------------------------------------------------------------
# Topic Explorer page keys
# ---------------------------------------------------------------------------

KEY_TOPIC_PREF_ANALYSIS_UNIT = "_topic_pref_analysis_unit"
KEY_TOPIC_PREF_MIN_CHARS = "_topic_pref_min_chars"
KEY_TOPIC_PREF_EMBEDDING_MODEL = "_topic_pref_embedding_model"
KEY_TOPIC_PREF_CUSTOM_MODEL = "_topic_pref_custom_model"
KEY_TOPIC_PREF_HIGHLEVEL = "_topic_pref_highlevel"
KEY_TOPIC_PREF_MIDLEVEL = "_topic_pref_midlevel"
KEY_TOPIC_PREF_LOWLEVEL = "_topic_pref_lowlevel"
KEY_TOPIC_ACTIVE_QUERY = "_topic_active_query"
KEY_TOPIC_ACTIVE_METHOD = "_topic_active_method"
KEY_TOPIC_STATE_KEY = "_topic_state_key"
KEY_TOPIC_DOCS = "_topic_docs"
KEY_TOPIC_EMBEDDINGS = "_topic_embeddings"
KEY_TOPIC_REDUCED = "_topic_reduced"
KEY_TOPIC_LABEL_LAYERS = "_topic_label_layers"
KEY_TOPIC_COUNTS = "_topic_counts"
KEY_TOPIC_MAP_SIG = "_topic_map_sig"
KEY_TOPIC_MAP_TYPE_PREF = "_topic_map_type_pref"
KEY_TOPIC_HAD_SCORE = "_topic_had_score"
# Widget keys also accessed directly in session state
WKEY_TOPIC_SEARCH_METHOD = "topic_search_method"
WKEY_TOPIC_SEARCH_QUERY = "topic_search_query"
WKEY_TOPIC_SORT_COL = "topic_sort_col"
WKEY_TOPIC_SORT_ASC = "topic_sort_asc"

# ---------------------------------------------------------------------------
# Document Terms page keys
# ---------------------------------------------------------------------------

KEY_DT_SPACY_MODEL = "_dt_spacy_model_pref"
KEY_DT_CACHE_KEY = "_doc_terms_key"
KEY_DT_DEFS = "_doc_terms_defs"
KEY_DT_DATES = "_doc_terms_dates"
KEY_DT_PARTIES = "_doc_terms_parties"
KEY_DT_MONEY = "_doc_terms_money"
KEY_DT_NUMBERS = "_doc_terms_numbers"

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
