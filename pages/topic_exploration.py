import io
import json
import math
import re
from collections.abc import Sequence

import datamapplot
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import Stemmer as _PyStemmer
from spacy.lang.en.stop_words import STOP_WORDS
from sentence_transformers import SentenceTransformer
from umap import UMAP

from src.comments.extract import extract_paragraphs


def _clean_docs(paragraphs: Sequence[str], min_chars: int) -> list[str]:
    docs: list[str] = []
    for paragraph in paragraphs:
        text = (paragraph or "").strip()
        if len(text) >= min_chars:
            docs.append(text)
    return docs


def _paragraphs_to_sentences(paragraphs: Sequence[str], min_chars: int) -> list[str]:
    sentences: list[str] = []
    for paragraph in paragraphs:
        text = (paragraph or "").strip()
        if not text:
            continue
        parts = re.split(r"(?<=[.!?])\s+", text)
        for part in parts:
            sent = part.strip()
            if len(sent) >= min_chars:
                sentences.append(sent)
    return sentences


def _topic_labels(topic_model: BERTopic, topics: list[int]) -> np.ndarray:
    labels: list[str] = []
    for topic in topics:
        if topic == -1:
            labels.append("Noise")
            continue
        words = topic_model.get_topic(topic) or []
        top_words = [word for word, _ in words[:3]]
        if top_words:
            labels.append(" ".join(top_words))
        else:
            labels.append(f"Topic {topic}")
    return np.array(labels, dtype=object)


@st.cache_resource(show_spinner=False)
def _get_encoder(embedding_model_name: str) -> SentenceTransformer:
    return SentenceTransformer(embedding_model_name)


@st.cache_resource(show_spinner=False)
def _embed_docs(docs: tuple[str, ...], embedding_model_name: str) -> np.ndarray:
    encoder = _get_encoder(embedding_model_name)
    return encoder.encode(list(docs), show_progress_bar=False)


@st.cache_resource(show_spinner=False)
def _fit_topics(
    docs: tuple[str, ...],
    embeddings: np.ndarray,
    min_topic_size: int,
    seed_topic_list: tuple[tuple[str, ...], ...] | None = None,
) -> tuple[BERTopic, list[int]]:
    model = BERTopic(
        min_topic_size=min_topic_size,
        vectorizer_model=CountVectorizer(stop_words="english"),
        seed_topic_list=[list(group) for group in seed_topic_list] if seed_topic_list else None,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(list(docs), embeddings)
    return model, topics


@st.cache_resource(show_spinner=False)
def _reduce_to_2d(embeddings: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    reducer = UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def _default_granularity(n_docs: int) -> tuple[int, int, int]:
    coarse = max(8, n_docs // 12)
    medium = max(5, n_docs // 20)
    fine = max(3, n_docs // 35)
    medium = min(medium, coarse)
    fine = min(fine, medium)
    return coarse, medium, fine


_TOPIC_PALETTE = [
    "#ffe066", "#b5ead7", "#b5d5ff", "#e8b5ff", "#ffb5b5",
    "#ffd9b5", "#b5ffe4", "#c9ffb5", "#ffb5e8", "#b5f0ff",
]


def _topic_color_map(label_layer: np.ndarray) -> dict[str, str]:
    unique = [l for l in dict.fromkeys(label_layer) if l != "Noise"]
    return {label: _TOPIC_PALETTE[i % len(_TOPIC_PALETTE)] for i, label in enumerate(unique)}


def _highlight_topic_keywords(text: str, topic_label: str, color: str) -> str:
    """Highlight words from the topic label in the text using a topic-specific color."""
    if not topic_label or topic_label == "Noise":
        return text
    stemmed_keywords = {
        _stemmer.stemWord(w.lower())
        for w in topic_label.split()
        if w.lower() not in STOP_WORDS and len(w) > 2
    }
    if not stemmed_keywords:
        return text

    def _replace(m: re.Match) -> str:
        word = m.group(0)
        if _stemmer.stemWord(word.lower()) in stemmed_keywords:
            return f'<mark style="background:{color}">{word}</mark>'
        return word

    return re.sub(r"\b\w+\b", _replace, text)


def _highlight_term(text: str, query: str) -> str:
    if not query.strip():
        return text
    pattern = re.compile(re.escape(query), flags=re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)


_stemmer = _PyStemmer.Stemmer("english")


def _highlight_query_tokens(text: str, query: str) -> str:
    stemmed_terms = {
        _stemmer.stemWord(term)
        for term in _tokenize(query)
        if term not in STOP_WORDS
    }
    if not stemmed_terms:
        return text

    def _replace(m: re.Match) -> str:
        word = m.group(0)
        if _stemmer.stemWord(word.lower()) in stemmed_terms:
            return f"<mark>{word}</mark>"
        return word

    return re.sub(r"\b\w+\b", _replace, text)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _bm25_scores(docs: Sequence[str], query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    query_terms = _tokenize(query)
    if not query_terms:
        return np.zeros(len(docs), dtype=float)

    tokenized_docs = [_tokenize(doc) for doc in docs]
    doc_lens = np.array([len(tokens) for tokens in tokenized_docs], dtype=float)
    avgdl = float(doc_lens.mean()) if len(doc_lens) else 0.0
    n_docs = len(docs)

    doc_freq: dict[str, int] = {}
    for tokens in tokenized_docs:
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    scores = np.zeros(n_docs, dtype=float)
    for idx, tokens in enumerate(tokenized_docs):
        tf: dict[str, int] = {}
        for term in tokens:
            tf[term] = tf.get(term, 0) + 1
        dl = max(doc_lens[idx], 1.0)
        for term in query_terms:
            if term not in tf:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            freq = tf[term]
            denom = freq + k1 * (1 - b + b * dl / max(avgdl, 1.0))
            scores[idx] += idf * (freq * (k1 + 1)) / max(denom, 1e-9)
    return scores


file_bytes = st.session_state.get("p1_file_bytes")
if not file_bytes:
    st.info("Upload a document on Page 1 to get started.")
    st.stop()

doc_paragraphs = extract_paragraphs(io.BytesIO(file_bytes))

st.sidebar.markdown("### Topic Extraction")
analysis_unit = st.sidebar.segmented_control(
    "Analysis unit",
    options=["Paragraph", "Sentence"],
    default="Paragraph",
    help=(
        "Paragraph is best for legal docs in most cases. It keeps context and yields "
        "cleaner, more stable topics. Sentence is useful for very fine-grained issue "
        "spotting but can be noisier."
    ),
)
min_chars = st.sidebar.slider(
    "Minimum text length",
    20,
    400,
    80,
    5,
    help="Short text can be noisy. Increase this to focus on richer text.",
)
embedding_model_name = st.sidebar.selectbox(
    "Embedding model",
    options=["all-mpnet-base-v2", "all-MiniLM-L6-v2",],
    index=0,
    help="This model converts each paragraph into vectors before topic clustering.",
)

if analysis_unit == "Sentence":
    docs = _paragraphs_to_sentences(doc_paragraphs.paragraphs, min_chars=min_chars)
else:
    docs = _clean_docs(doc_paragraphs.paragraphs, min_chars=min_chars)

if len(docs) < 12:
    st.warning(
        "Not enough text after filtering. Lower 'Minimum text length' or use a longer document."
    )
    st.stop()

st.subheader("Topic Explorer")
st.markdown("#### Search")
search_query = st.text_input(
    "Filter text",
    label_visibility="collapsed",
    key="p3_search_query",
    placeholder="Search across all topics...",
)
search_method = st.pills(
    "Search type",
    options=["Keyword", "Regex", "Relevance", "Semantic"],
    default="Keyword",
    key="p3_search_method",
    label_visibility="collapsed",
    selection_mode="single",
) or "Keyword"
if search_method != "Keyword":
    st.sidebar.markdown("### Search")
    rank_limit = st.sidebar.slider(
        "Result limit",
        min_value=10,
        max_value=500,
        value=200,
        step=10,
        key="p3_rank_limit",
    )
    if search_method == "Semantic":
        semantic_min_score = st.sidebar.slider(
            "Minimum score",
            min_value=-1.0,
            max_value=1.0,
            value=0.20,
            step=0.01,
            key="p3_semantic_min_score",
            help="Higher values return stricter matches.",
        )
    else:
        semantic_min_score = st.session_state.get("p3_semantic_min_score", 0.20)
else:
    rank_limit = st.session_state.get("p3_rank_limit", 200)
    semantic_min_score = st.session_state.get("p3_semantic_min_score", 0.20)


normalized_query = search_query.strip().lower()
coarse_default, medium_default, fine_default = _default_granularity(len(docs))
st.sidebar.markdown("### Topics (Broad to Detailed)")
st.sidebar.caption(
    "Adjust topic level granularity.\n"
    "- Higher value: fewer, broader topics.\n"
    "- Lower value: more, narrower topics."
)
coarse_size = st.sidebar.slider(
    "Broad topics",
    2,
    max(2, min(50, len(docs) // 4)),
    coarse_default,
    help="Start here for high-level topics. Increase to merge topics into larger groups.",
)
medium_size = st.sidebar.slider(
    "Mid-level topics",
    2,
    max(2, min(30, len(docs) // 6)),
    medium_default,
    help="Balances specificity and stability. Usually between broad and detailed topic values.",
)
fine_size = st.sidebar.slider(
    "Detailed topics",
    2,
    max(2, min(15, len(docs) // 10)),
    fine_default,
    help="Use lower values to surface niche sub-topics. Too low can produce noisy clusters.",
)

granularity_sizes = sorted({coarse_size, medium_size, fine_size}, reverse=True)

with st.sidebar.expander("Seed words (advanced)", expanded=False):
    st.caption(
        "Guide topics by entering seed words — one group per line, words separated by commas.\n\n"
        "Example:\n```\nprivacy, personal data, consent\nliability, damages\ntermination, expiry\n```"
    )
    seed_words_raw = st.text_area(
        "Seed words",
        label_visibility="collapsed",
        key="p3_seed_words",
        placeholder="privacy, personal data, consent\nliability, damages\ntermination, expiry",
        height=120,
    )

seed_topic_list: tuple[tuple[str, ...], ...] | None = None
if seed_words_raw and seed_words_raw.strip():
    parsed = tuple(
        tuple(w.strip() for w in line.split(",") if w.strip())
        for line in seed_words_raw.strip().splitlines()
        if line.strip()
    )
    if parsed:
        seed_topic_list = parsed

with st.spinner("Embedding and modeling topics..."):
    docs_tuple = tuple(docs)
    embeddings = _embed_docs(docs_tuple, embedding_model_name)
    reduced_embeddings = _reduce_to_2d(
        embeddings=embeddings,
        n_neighbors=min(30, max(5, len(docs) // 25)),
        min_dist=0.05,
    )

    label_layers: list[np.ndarray] = []
    topic_counts: list[int] = []
    for min_topic_size in granularity_sizes:
        topic_model, topics = _fit_topics(docs_tuple, embeddings, min_topic_size, seed_topic_list)
        labels = _topic_labels(topic_model, topics)
        label_layers.append(labels)
        n_topics = len({t for t in topics if t != -1})
        topic_counts.append(n_topics)

st.sidebar.markdown("### Topics")
sidebar_stats_cols = st.sidebar.columns(len(granularity_sizes))
for idx, size in enumerate(granularity_sizes):
    topic_name = (
        "Broad"
        if idx == 0
        else "Mid"
        if idx == 1
        else "Detailed"
    )
    sidebar_stats_cols[idx].metric(topic_name, topic_counts[idx])

has_noise = any((labels == "Noise").any() for labels in label_layers)
if has_noise:
    st.sidebar.caption(
        "Noise: text that doesn't cluster into any topic. "
        "Raise minimum text length or adjust granularity to reduce it."
    )

score_map: dict[int, float] = {}
if not normalized_query:
    matched_indices = list(range(len(docs)))
elif search_method == "Keyword":
    matched_indices = [idx for idx, text in enumerate(docs) if normalized_query in text.lower()]
elif search_method == "Regex":
    try:
        pattern = re.compile(search_query.strip(), flags=re.IGNORECASE)
        matched_indices = [idx for idx, text in enumerate(docs) if pattern.search(text)]
    except re.error as e:
        st.warning(f"Invalid regex: {e}")
        matched_indices = []
elif search_method == "Relevance":
    bm25_scores = _bm25_scores(docs, normalized_query)
    ranked = np.argsort(-bm25_scores)
    matched_indices = [int(i) for i in ranked if bm25_scores[i] > 0][:rank_limit]
    score_map = {int(i): float(bm25_scores[i]) for i in matched_indices}
else:
    encoder = _get_encoder(embedding_model_name)
    query_embedding = encoder.encode([normalized_query], show_progress_bar=False)[0]
    doc_norms = np.linalg.norm(embeddings, axis=1)
    query_norm = float(np.linalg.norm(query_embedding))
    cosine_scores = (embeddings @ query_embedding) / np.clip(doc_norms * query_norm, 1e-9, None)
    ranked = np.argsort(-cosine_scores)
    matched_indices = [int(i) for i in ranked if cosine_scores[i] >= semantic_min_score][:rank_limit]
    score_map = {int(i): float(cosine_scores[i]) for i in matched_indices}

if len(matched_indices) >= 10:
    plot_embeddings = reduced_embeddings[matched_indices]
    plot_docs = [docs[idx] for idx in matched_indices]
    plot_label_layers = [labels[matched_indices] for labels in label_layers]

    extra_data = pd.DataFrame({"paragraph_idx": matched_indices})

    plot = datamapplot.create_interactive_plot(
        plot_embeddings,
        font_family="Arial",
        *plot_label_layers,
        hover_text=np.array(plot_docs, dtype=object),
        extra_point_data=extra_data,
        on_click="openDoc(hoverData.paragraph_idx[index]);",
        enable_search=False,
        sub_title="Click a point to open the passage in context. Zoom in to reveal finer topic labels.",
        # width="100%",
        # height=820,
        initial_zoom_fraction=0.33,
        cluster_boundary_polygons=True,
        cluster_boundary_line_width=6,
    )

    all_docs_json = json.dumps(list(docs))
    inject_script = f"""<script>
window.__ALL_DOCS__ = {all_docs_json};
window.openDoc = function(paraIdx) {{
  var allDocs = window.__ALL_DOCS__;
  var rows = allDocs.map(function(text, i) {{
    var safe = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    var idAttr = i === paraIdx ? 'id="active"' : '';
    var cls = i === paraIdx ? ' class="active"' : '';
    return '<p ' + idAttr + cls + '>' + safe + '</p>';
  }});
  var css = '<style>body{{font-family:Georgia,serif;max-width:900px;margin:40px auto;padding:0 20px;line-height:1.8;color:#333}}p{{margin:1em 0;padding:4px 0}}.active{{background:#fffde7;border-left:4px solid #f9a825;padding:.5em 1em;border-radius:0 4px 4px 0}}</style>';
  var html = '<!DOCTYPE html><html><head><meta charset="utf-8">'+css+'</head><body>'+rows.join('')+'</body></html>';
  var blob = new Blob([html],{{type:'text/html'}});
  window.open(URL.createObjectURL(blob)+'#active','_blank');
}};
</script>"""

    plot_html = str(plot).replace("</body>", inject_script + "</body>", 1)

    st.markdown("#### Map")
    expanded = len(matched_indices) >= 100
    with st.expander("Topic map", expanded=expanded):
        components.html(plot_html, height=860, scrolling=False)
elif matched_indices:
    st.info("Too few matches to render a map (minimum 10). Results are shown in the table below.")
else:
    st.info("No text matches your search. Try a broader term.")

max_rows = st.sidebar.slider("Max rows", min_value=10, max_value=500, value=100, step=10, key="p3_max_rows")

results_df = pd.DataFrame(
    {
        "paragraph_idx": np.arange(len(docs)),
        "text": docs,
    }
)
topic_columns = ["Broad topics", "Mid-level topics", "Detailed topics"]
for idx, size in enumerate(granularity_sizes):
    topic_col_name = topic_columns[idx] if idx < len(topic_columns) else f"Topics {idx + 1}"
    results_df[topic_col_name] = label_layers[idx]

results_df = results_df.iloc[matched_indices].reset_index(drop=True) if matched_indices else results_df.iloc[0:0]
if score_map:
    results_df["score"] = results_df["paragraph_idx"].map(score_map)
    results_df = results_df.sort_values("score", ascending=False).reset_index(drop=True)

st.divider()
st.markdown("#### Filter")
available_topic_cols = [col for col in topic_columns if col in results_df.columns]
if available_topic_cols and not results_df.empty:
    fcol1, fcol2 = st.columns([1, 3])
    filter_col = fcol1.selectbox(
        "Granularity",
        options=available_topic_cols,
        index=len(available_topic_cols) - 1,
        key="p3_topic_filter_col",
        label_visibility="collapsed",
    )
    topic_options = sorted(t for t in results_df[filter_col].unique() if t != "Noise")
    selected_topics = fcol2.multiselect(
        "Topics",
        options=topic_options,
        default=[],
        key="p3_topic_filter_values",
        placeholder=f"Select {filter_col.lower()}…",
        label_visibility="collapsed",
    )
    if selected_topics:
        results_df = results_df[results_df[filter_col].isin(selected_topics)].reset_index(drop=True)

st.markdown("#### Results")
st.caption(f"{len(results_df)} passages")
st.dataframe(
    results_df.head(max_rows),
    width="stretch",
    hide_index=True,
    column_config={
        "paragraph_idx": st.column_config.NumberColumn("#", width="small"),
        "text": st.column_config.TextColumn("Passage", width="large"),
        "score": st.column_config.NumberColumn("Score", format="%.4f", width="small"),
    },
)

show_markdown = st.checkbox(
    "Expand results",
    key="p3_print_markdown",
    value=False,
)

if show_markdown:
    shown_df = results_df.head(max_rows).copy()
    topic_cols = [col for col in topic_columns if col in shown_df.columns]
    if shown_df.empty:
        st.markdown("_No matching rows._")
    else:
        finest_labels = label_layers[-1]
        color_map = _topic_color_map(finest_labels)
        lines: list[str] = []
        for _, row in shown_df.iterrows():
            para_idx = int(row["paragraph_idx"])
            text = str(row["text"])
            # Search query highlighting
            if search_method in ("Relevance", "Semantic"):
                highlighted = _highlight_query_tokens(text, search_query)
            elif search_method == "Regex":
                try:
                    highlighted = re.compile(search_query.strip(), flags=re.IGNORECASE).sub(
                        lambda m: f"<mark>{m.group(0)}</mark>", text
                    )
                except re.error:
                    highlighted = text
            else:
                highlighted = _highlight_term(text, search_query)
            # Topic keyword highlighting (on top of search highlighting)
            topic_label = finest_labels[para_idx] if para_idx < len(finest_labels) else ""
            color = color_map.get(str(topic_label), "#ffe066")
            highlighted = _highlight_topic_keywords(highlighted, str(topic_label), color)
            topics = " → ".join(f"{row[col]}" for col in topic_cols)
            lines.append(
                f"#### ¶{para_idx} — {topics}\n\n"
                f"{highlighted}"
            )
        st.markdown("\n\n---\n\n".join(lines), unsafe_allow_html=True)
