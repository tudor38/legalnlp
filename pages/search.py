"""
Search — query across multiple uploaded Word documents.

Supports three search methods:
  Keyword  — exact substring match (case-insensitive)
  Relevance — BM25 ranking across all documents
  Semantic  — cosine similarity via sentence-transformers embeddings
"""

import io
import re
from collections.abc import Sequence

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.comments.extract import extract_paragraphs


# ---------------------------------------------------------------------------
# Search helpers  (BM25 + tokenizer lifted from topic_exploration.py)
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _bm25_scores(docs: Sequence[str], query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    query_terms = _tokenize(query)
    if not query_terms:
        return np.zeros(len(docs), dtype=float)

    tokenized_docs = [_tokenize(doc) for doc in docs]
    doc_lens = np.array([len(t) for t in tokenized_docs], dtype=float)
    avgdl = float(doc_lens.mean()) if len(doc_lens) else 1.0
    n_docs = len(docs)

    doc_freq: dict[str, int] = {}
    for tokens in tokenized_docs:
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    scores = np.zeros(n_docs, dtype=float)
    for term in query_terms:
        df = doc_freq.get(term, 0)
        if df == 0:
            continue
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        for i, tokens in enumerate(tokenized_docs):
            tf = tokens.count(term)
            denom = tf + k1 * (1 - b + b * doc_lens[i] / avgdl)
            scores[i] += idf * (tf * (k1 + 1)) / denom

    return scores


def _keyword_highlight(text: str, query: str) -> str:
    if not query:
        return text
    return re.sub(f"({re.escape(query)})", r"<mark>\1</mark>", text, flags=re.IGNORECASE)


def _regex_highlight(text: str, pattern: re.Pattern) -> str:
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)


# ---------------------------------------------------------------------------
# Cached models + extraction
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_encoder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_data(show_spinner=False)
def _extract(file_bytes: bytes) -> list[str]:
    doc = extract_paragraphs(io.BytesIO(file_bytes))
    return [p.strip() for p in doc.paragraphs if len(p.strip()) >= 30]


@st.cache_data(show_spinner=False)
def _embed(texts: tuple[str, ...], model_name: str) -> np.ndarray:
    encoder = _get_encoder(model_name)
    return encoder.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.subheader("Search")

uploaded = st.file_uploader(
    "Upload documents",
    type=["docx", "doc"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not uploaded:
    st.caption("Upload one or more Word documents to search across them.")
    st.stop()

# Sidebar
st.sidebar.markdown("### Search")
method = st.sidebar.radio(
    "Method",
    ["Keyword", "Regex", "Relevance", "Semantic"],
    index=2,
)
if method == "Semantic":
    model_name = st.sidebar.selectbox(
        "Embedding model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        index=0,
    )
max_results = st.sidebar.slider("Max results", 5, 50, 20)

# Build corpus: list of (doc_name, para_text)
corpus: list[tuple[str, str]] = []
for f in uploaded:
    paras = _extract(f.getvalue())
    for p in paras:
        corpus.append((f.name, p))

doc_names = [name for name, _ in corpus]
texts = [text for _, text in corpus]

total_docs = len({n for n in doc_names})
st.caption(f"{total_docs} document{'s' if total_docs != 1 else ''} · {len(texts)} passages")

# Search input
query = st.text_input(
    "Query",
    placeholder="Search across all documents…",
    label_visibility="collapsed",
)

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
        regex_error = None
    except re.error as e:
        st.warning(f"Invalid regex: {e}")
        st.stop()
    hits = [(i, 1.0) for i, t in enumerate(texts) if pattern.search(t)][:max_results]

elif method == "Relevance":
    scores = _bm25_scores(texts, q)
    ranked = np.argsort(-scores)
    hits = [(int(i), float(scores[i])) for i in ranked if scores[i] > 0][:max_results]

else:  # Semantic
    with st.spinner("Embedding…"):
        corpus_embs = _embed(tuple(texts), model_name)
        encoder = _get_encoder(model_name)
        q_emb = encoder.encode([q], show_progress_bar=False, normalize_embeddings=True)[0]
    sims = corpus_embs @ q_emb
    ranked = np.argsort(-sims)
    hits = [(int(i), float(sims[i])) for i in ranked[:max_results]]

if not hits:
    st.info("No matches found.")
    st.stop()

st.caption(f"{len(hits)} result{'s' if len(hits) != 1 else ''}")

# Results
for idx, score in hits:
    doc_name = doc_names[idx]
    passage = texts[idx]

    if method == "Keyword":
        display = _keyword_highlight(passage, q)
        score_str = ""
    elif method == "Regex":
        display = _regex_highlight(passage, pattern)
        score_str = ""
    elif method == "Relevance":
        display = _keyword_highlight(passage, q)
        score_str = f"{score:.2f}"
    else:
        display = passage
        score_str = f"{score:.2f}"

    with st.container(border=True):
        col_doc, col_score = st.columns([5, 1])
        with col_doc:
            st.caption(doc_name)
        with col_score:
            if score_str:
                st.caption(score_str)
        st.markdown(display, unsafe_allow_html=True)
