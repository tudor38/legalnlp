"""
Search — query across multiple uploaded Word documents.

Supports three search methods:
  Keyword  — exact substring match (case-insensitive)
  Regex    — regular expression match
  Relevance — BM25 ranking across all documents
  Semantic  — cosine similarity via sentence-transformers embeddings
"""

import io
import re

import numpy as np
import streamlit as st

from src.app_state import MODEL_MINILM, MODEL_MPNET, get_file_bytes, get_file_name
from src.comments.extract import extract_paragraphs
from src.shared import DocxParseError
from src.utils.models import get_sentence_transformer
from src.utils.text import (
    TOPIC_PALETTE,
    bm25_scores,
    highlight_query_tokens,
    highlight_term,
)


# ---------------------------------------------------------------------------
# Cached extraction
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=10)
def _extract(file_bytes: bytes) -> list[str]:
    doc = extract_paragraphs(io.BytesIO(file_bytes))
    return [p.strip() for p in doc.paragraphs if len(p.strip()) >= 30]


@st.cache_data(show_spinner=False, max_entries=10)
def _embed(texts: tuple[str, ...], model_name: str) -> np.ndarray:
    encoder = get_sentence_transformer(model_name)
    return encoder.encode(
        list(texts), show_progress_bar=False, normalize_embeddings=True
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.subheader("Search")

global_bytes = get_file_bytes()
global_name = get_file_name()

# Sidebar — file upload
if global_name:
    st.sidebar.caption(f"📄 {global_name}")

uploaded = st.sidebar.file_uploader(
    "Add more files" if global_name else "Upload documents",
    type=["docx", "doc"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

# Persist extra uploaded files across navigations
if uploaded:
    st.session_state["_search_stored_files"] = [
        (f.name, f.getvalue()) for f in uploaded
    ]

stored_extra: list[tuple[str, bytes]] = st.session_state.get("_search_stored_files", [])

# Build corpus sources
extra_files = [(f.name, f.getvalue()) for f in uploaded] if uploaded else stored_extra

files_to_use: list[tuple[str, bytes]] = []
if global_bytes and global_name:
    files_to_use.append((global_name, global_bytes))
files_to_use.extend(extra_files)

if not files_to_use:
    st.caption("Upload one or more Word documents to search across them.")
    st.stop()

if not uploaded and stored_extra:
    names = ", ".join(n for n, _ in stored_extra)
    st.sidebar.caption(f"Also: {names}")
    if st.sidebar.button("Clear extra files", key="search_clear_files"):
        del st.session_state["_search_stored_files"]
        st.rerun()

# Sidebar — per-page preference
_saved_per_page = st.session_state.get("_search_pref_per_page", 10)
per_page = st.sidebar.selectbox(
    "Results per page",
    options=[5, 10, 20, 50],
    index=[5, 10, 20, 50].index(_saved_per_page)
    if _saved_per_page in [5, 10, 20, 50]
    else 1,
)
st.session_state["_search_pref_per_page"] = per_page

# Build corpus: list of (doc_name, para_text)
corpus: list[tuple[str, str]] = []
for name, file_bytes in files_to_use:
    try:
        paras = _extract(file_bytes)
    except DocxParseError:
        st.warning(f"'{name}' is not a valid Word document and was skipped.")
        continue
    for p in paras:
        corpus.append((name, p))

doc_names = [name for name, _ in corpus]
texts = [text for _, text in corpus]

total_docs = len({n for n in doc_names})
st.caption(
    f"{total_docs} document{'s' if total_docs != 1 else ''} · {len(texts)} passages"
)

# Search input
_saved_query = st.session_state.get("_search_pref_query", "")
query = st.text_input(
    "Query",
    value=_saved_query,
    placeholder="Search across all documents…",
    label_visibility="collapsed",
    key="search_query",
)
st.session_state["_search_pref_query"] = query

_method_options = ["Keyword", "Regex", "Relevance", "Semantic"]
st.session_state.setdefault("_search_pref_method", "Keyword")
st.session_state.setdefault("search_method", st.session_state["_search_pref_method"])
method = (
    st.pills(
        "Search type",
        options=_method_options,
        key="search_method",
        label_visibility="collapsed",
        selection_mode="single",
    )
    or "Keyword"
)
st.session_state["_search_pref_method"] = method

min_score: float = 0.0
if method in ("Relevance", "Semantic"):
    _saved_min_score = st.session_state.get("_search_pref_min_score", 0.0)
    min_score = st.sidebar.slider(
        "Minimum score",
        min_value=0.0,
        max_value=1.0,
        value=_saved_min_score,
        step=0.01,
        key="search_min_score",
    )
    st.session_state["_search_pref_min_score"] = min_score

_model_options = [MODEL_MINILM, MODEL_MPNET]
if method == "Semantic":
    _saved_model = st.session_state.get("_search_pref_model")
    _selected = st.sidebar.selectbox(
        "Embedding model",
        _model_options + ["Custom…"],
        index=_model_options.index(_saved_model)
        if _saved_model in _model_options
        else 0,
        key="search_model",
    )
    if _selected == "Custom…":
        model_name = st.sidebar.text_input(
            "HuggingFace model ID",
            value=st.session_state.get("_search_pref_custom_model", ""),
            placeholder="e.g. BAAI/bge-small-en-v1.5",
            key="search_custom_model",
        ).strip()
        st.session_state["_search_pref_custom_model"] = model_name
        if not model_name:
            st.info(
                "Enter a HuggingFace model ID above to use a custom embedding model."
            )
            st.stop()
    else:
        model_name = _selected
    st.session_state["_search_pref_model"] = model_name

if not query or not query.strip():
    st.stop()

q = query.strip()

# Build a key that uniquely identifies this search. Hits are cached in session
# state so pagination never re-runs the search.
_score_key = round(min_score, 2) if method in ("Relevance", "Semantic") else 0.0
_model_key = model_name if method == "Semantic" else ""
_corpus_hash = hash(tuple(texts))
search_key = (q, method, _score_key, _model_key, _corpus_hash)

if st.session_state.get("_search_hits_key") != search_key:
    # Parameters changed — run the search and cache results.
    if method == "Keyword":
        pattern = re.compile(re.escape(q), re.IGNORECASE)
        hits = [(i, 1.0) for i, t in enumerate(texts) if pattern.search(t)]

    elif method == "Regex":
        try:
            pattern = re.compile(q, re.IGNORECASE)
        except re.error as e:
            st.warning(f"Invalid regex: {e}")
            st.stop()
        hits = [(i, 1.0) for i, t in enumerate(texts) if pattern.search(t)]

    elif method == "Relevance":
        scores = bm25_scores(texts, q)
        ranked = np.argsort(-scores)
        all_positive = [(int(i), float(scores[i])) for i in ranked if scores[i] > 0]
        hits = [(i, s) for i, s in all_positive if s >= min_score]
        if not hits and all_positive:
            st.info(
                f"No results above the minimum score ({min_score:.2f}). Try lowering it in the sidebar."
            )
            st.stop()

    else:  # Semantic
        try:
            with st.spinner("Embedding…"):
                corpus_embs = _embed(tuple(texts), model_name)
                encoder = get_sentence_transformer(model_name)
                q_emb = encoder.encode(
                    [q], show_progress_bar=False, normalize_embeddings=True
                )[0]
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
        sims = corpus_embs @ q_emb
        ranked = np.argsort(-sims)
        all_ranked = [(int(i), float(sims[i])) for i in ranked]
        hits = [(i, s) for i, s in all_ranked if s >= min_score]
        if not hits and any(s > 0 for _, s in all_ranked):
            st.info(
                f"No results above the minimum score ({min_score:.2f}). Try lowering it in the sidebar."
            )
            st.stop()

    st.session_state["_search_hits"] = hits
    st.session_state["_search_hits_key"] = search_key
    st.session_state["_search_page"] = 0
else:
    hits = st.session_state["_search_hits"]

if not hits:
    st.info("No matches found.")
    st.stop()

total_hits = len(hits)
total_pages = max(1, -(-total_hits // per_page))  # ceiling division

page = st.session_state.get("_search_page", 0)
page = max(0, min(page, total_pages - 1))

page_hits = hits[page * per_page : (page + 1) * per_page]

col_count, col_prev, col_page, col_next = st.columns([4, 1, 2, 1])
col_count.caption(f"{total_hits} result{'s' if total_hits != 1 else ''}")
if total_pages > 1:
    if col_prev.button("← Prev", disabled=page == 0, key="search_prev"):
        st.session_state["_search_page"] = page - 1
        st.rerun()
    col_page.caption(
        f"Page {page + 1} of {total_pages}",
    )
    if col_next.button("Next →", disabled=page >= total_pages - 1, key="search_next"):
        st.session_state["_search_page"] = page + 1
        st.rerun()

# Assign a color to each unique document name
unique_docs = list(dict.fromkeys(doc_names))
doc_color = {
    name: TOPIC_PALETTE[i % len(TOPIC_PALETTE)] for i, name in enumerate(unique_docs)
}

# Results
for idx, score in page_hits:
    doc_name = doc_names[idx]
    passage = texts[idx]
    color = doc_color[doc_name]

    if method == "Keyword":
        display = highlight_term(passage, q, color)
        score_str = ""
    elif method == "Regex":
        display = pattern.sub(
            lambda m, c=color: f'<mark style="background:{c}">{m.group(0)}</mark>',
            passage,
        )
        score_str = ""
    elif method == "Relevance":
        display = highlight_query_tokens(passage, q, color)
        score_str = f"{score:.2f}"
    else:  # Semantic
        display = highlight_query_tokens(passage, q, color)
        score_str = f"{score:.2f}"

    with st.container(border=True):
        col_doc, col_score = st.columns([5, 1])
        with col_doc:
            st.markdown(
                f'<span style="background:{color};padding:2px 8px;border-radius:4px;font-size:0.8em">{doc_name}</span>',
                unsafe_allow_html=True,
            )
        with col_score:
            if score_str:
                st.caption(score_str)
        st.markdown(display, unsafe_allow_html=True)
