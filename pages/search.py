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
import Stemmer as _PyStemmer
from spacy.lang.en.stop_words import STOP_WORDS

from src.app_state import MODEL_MINILM, MODEL_MPNET, get_file_bytes, get_file_name
from src.comments.extract import extract_paragraphs
from src.shared import DocxParseError
from src.utils.models import get_sentence_transformer
from src.utils.text import TOPIC_PALETTE, bm25_scores, tokenize


_stemmer = _PyStemmer.Stemmer("english")


def _highlight_term(text: str, query: str, color: str = "") -> str:
    if not query.strip():
        return text
    style = f' style="background:{color}"' if color else ""
    return re.compile(re.escape(query), flags=re.IGNORECASE).sub(
        lambda m: f"<mark{style}>{m.group(0)}</mark>", text
    )


def _highlight_query_tokens(text: str, query: str, color: str = "") -> str:
    stemmed_terms = {
        _stemmer.stemWord(term) for term in tokenize(query) if term not in STOP_WORDS
    }
    if not stemmed_terms:
        return text
    style = f' style="background:{color}"' if color else ""

    def _replace(m: re.Match) -> str:
        word = m.group(0)
        if _stemmer.stemWord(word.lower()) in stemmed_terms:
            return f"<mark{style}>{word}</mark>"
        return word

    return re.sub(r"\b\w+\b", _replace, text)


# ---------------------------------------------------------------------------
# Cached extraction
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _extract(file_bytes: bytes) -> list[str]:
    doc = extract_paragraphs(io.BytesIO(file_bytes))
    return [p.strip() for p in doc.paragraphs if len(p.strip()) >= 30]


@st.cache_data(show_spinner=False)
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

if global_name:
    st.caption(f"📄 {global_name}")

uploaded = st.file_uploader(
    "Add more files" if global_name else "Upload documents",
    type=["docx", "doc"],
    accept_multiple_files=True,
    label_visibility="collapsed" if global_name else "collapsed",
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
    col_info, col_clear = st.columns([6, 1])
    col_info.caption(f"Also using: {names}")
    if col_clear.button("Clear", key="search_clear_files"):
        del st.session_state["_search_stored_files"]
        st.rerun()

# Sidebar
_saved_max = st.session_state.get("_search_pref_max_results", 20)
max_results = st.sidebar.slider("Max results", 5, 50, _saved_max)
st.session_state["_search_pref_max_results"] = max_results

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
method = st.pills(
    "Search type",
    options=_method_options,
    key="search_method",
    label_visibility="collapsed",
    selection_mode="single",
) or "Keyword"
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
    model_name = st.sidebar.selectbox(
        "Embedding model",
        _model_options,
        index=_model_options.index(_saved_model) if _saved_model in _model_options else 0,
        key="search_model",
    )
    st.session_state["_search_pref_model"] = model_name

if not query or not query.strip():
    st.stop()

q = query.strip()

# Run search
if method == "Keyword":
    pattern = re.compile(re.escape(q), re.IGNORECASE)
    hits = [(i, 1.0) for i, t in enumerate(texts) if pattern.search(t)][:max_results]

elif method == "Regex":
    try:
        pattern = re.compile(q, re.IGNORECASE)
    except re.error as e:
        st.warning(f"Invalid regex: {e}")
        st.stop()
    hits = [(i, 1.0) for i, t in enumerate(texts) if pattern.search(t)][:max_results]

elif method == "Relevance":
    scores = bm25_scores(texts, q)
    ranked = np.argsort(-scores)
    hits = [(int(i), float(scores[i])) for i in ranked if scores[i] >= min_score and scores[i] > 0][:max_results]

else:  # Semantic
    with st.spinner("Embedding…"):
        corpus_embs = _embed(tuple(texts), model_name)
        encoder = get_sentence_transformer(model_name)
        q_emb = encoder.encode([q], show_progress_bar=False, normalize_embeddings=True)[0]
    sims = corpus_embs @ q_emb
    ranked = np.argsort(-sims)
    hits = [(int(i), float(sims[i])) for i in ranked if sims[i] >= min_score][:max_results]

if not hits:
    st.info("No matches found.")
    st.stop()

st.caption(f"{len(hits)} result{'s' if len(hits) != 1 else ''}")

# Assign a color to each unique document name
unique_docs = list(dict.fromkeys(doc_names))
doc_color = {name: TOPIC_PALETTE[i % len(TOPIC_PALETTE)] for i, name in enumerate(unique_docs)}

# Results
for idx, score in hits:
    doc_name = doc_names[idx]
    passage = texts[idx]
    color = doc_color[doc_name]

    if method == "Keyword":
        display = _highlight_term(passage, q, color)
        score_str = ""
    elif method == "Regex":
        display = pattern.sub(lambda m, c=color: f'<mark style="background:{c}">{m.group(0)}</mark>', passage)
        score_str = ""
    elif method == "Relevance":
        display = _highlight_query_tokens(passage, q, color)
        score_str = f"{score:.2f}"
    else:  # Semantic
        display = _highlight_query_tokens(passage, q, color)
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
