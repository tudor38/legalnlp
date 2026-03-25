import base64
import hashlib
import io
import json
import re
import time
from collections.abc import Sequence

import datamapplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import Stemmer as _PyStemmer
from spacy.lang.en.stop_words import STOP_WORDS
from umap import UMAP

from src.app_state import KEY_TOPIC_RANK_LIMIT, KEY_TOPIC_SEMANTIC_MIN, MODEL_MINILM, MODEL_MPNET
from src.comments.extract import extract_paragraphs
from src.utils.models import get_sentence_transformer
from src.utils.page import require_document
from src.utils.text import TOPIC_PALETTE, bm25_scores, tokenize


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
def _embed_docs(docs: tuple[str, ...], embedding_model_name: str) -> np.ndarray:
    encoder = get_sentence_transformer(embedding_model_name)
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
        seed_topic_list=[list(group) for group in seed_topic_list]
        if seed_topic_list
        else None,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(list(docs), embeddings)
    return model, topics


@st.cache_resource(show_spinner=False)
def _reduce_to_2d(
    embeddings: np.ndarray, n_neighbors: int, min_dist: float
) -> np.ndarray:
    reducer = UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


@st.cache_data(show_spinner=False)
def _render_static_map(
    plot_embeddings: np.ndarray,
    plot_label_layers: tuple,
) -> str:
    """Return base64-encoded PNG."""
    finest = plot_label_layers[-1]
    fig, _ = datamapplot.create_plot(
        plot_embeddings,
        finest,
        noise_label="Noise",
        noise_color="#cccccc",
        dynamic_label_size=False,
        figsize=(22, 15),
        label_wrap_width=20,
        dpi=180,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


@st.cache_data(show_spinner=False)
def _render_interactive_map(
    plot_embeddings: np.ndarray,
    plot_label_layers: tuple,
    all_docs: tuple[str, ...],
    matched_indices: tuple[int, ...],
    zoom: float,
) -> str | None:
    """Return HTML string, or None if datamapplot cannot render this dataset interactively.

    datamapplot crashes when only one unique topic label exists: it produces a
    1-D label_locations array ([x, y]) instead of 2-D ([[x, y], ...]), causing
    an IndexError on label_locations[:, 0].  Guard against it up front.
    """
    finest = plot_label_layers[-1]
    n_unique = len({lbl for lbl in finest if lbl != "Noise"})
    if n_unique < 2:
        return None

    try:
        plot_docs = [all_docs[i] for i in matched_indices]
        plot = datamapplot.create_interactive_plot(
            plot_embeddings,
            *plot_label_layers,
            hover_text=np.array(plot_docs, dtype=object),
            extra_point_data=pd.DataFrame({"paragraph_idx": list(matched_indices)}),
            on_click="openDoc(hoverData.paragraph_idx[index]);",
            enable_search=False,
            noise_label="Noise",
            noise_color="#cccccc",
            sub_title="Click a point to open the passage in context. Zoom in to reveal finer topic labels.",
            initial_zoom_fraction=zoom,
        )
        all_docs_json = json.dumps(list(all_docs))
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
        return str(plot).replace("</body>", inject_script + "</body>", 1)
    except (IndexError, ValueError):
        return None


def _default_granularity(n_docs: int) -> tuple[int, int, int]:
    coarse = max(max(2, n_docs // 6), n_docs // 40)
    medium = max(max(2, n_docs // 9), n_docs // 80)
    fine = max(2, n_docs // 150)
    medium = min(medium, coarse)
    fine = min(fine, medium)
    return coarse, medium, fine


def _topic_color_map(label_layer: np.ndarray) -> dict[str, str]:
    unique = [l for l in dict.fromkeys(label_layer) if l != "Noise"]
    return {
        label: TOPIC_PALETTE[i % len(TOPIC_PALETTE)] for i, label in enumerate(unique)
    }


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
        _stemmer.stemWord(term) for term in tokenize(query) if term not in STOP_WORDS
    }
    if not stemmed_terms:
        return text

    def _replace(m: re.Match) -> str:
        word = m.group(0)
        if _stemmer.stemWord(word.lower()) in stemmed_terms:
            return f"<mark>{word}</mark>"
        return word

    return re.sub(r"\b\w+\b", _replace, text)


file_bytes = require_document()

doc_paragraphs = extract_paragraphs(io.BytesIO(file_bytes))

_model_options = [MODEL_MPNET, MODEL_MINILM]

st.sidebar.markdown("### Topic Extraction")
analysis_unit = st.sidebar.segmented_control(
    "Analysis unit",
    options=["Paragraph", "Sentence"],
    default=st.session_state.get("_topic_pref_analysis_unit", "Paragraph"),
    key="topic_analysis_unit",
    help=(
        "Paragraph is best for legal docs in most cases. It keeps context and yields "
        "cleaner, more stable topics. Sentence is useful for very fine-grained issue "
        "spotting but can be noisier."
    ),
)
st.session_state["_topic_pref_analysis_unit"] = analysis_unit

min_chars = st.sidebar.slider(
    "Minimum text length",
    20,
    400,
    st.session_state.get("_topic_pref_min_chars", 80),
    5,
    key="topic_min_chars",
    help="Short text can be noisy. Increase this to focus on richer text.",
)
st.session_state["_topic_pref_min_chars"] = min_chars

_saved_model = st.session_state.get("_topic_pref_embedding_model")
embedding_model_name = st.sidebar.selectbox(
    "Embedding model",
    options=_model_options,
    index=_model_options.index(_saved_model) if _saved_model in _model_options else 0,
    key="topic_embedding_model",
    help="This model converts each paragraph into vectors before topic clustering.",
)
st.session_state["_topic_pref_embedding_model"] = embedding_model_name

if analysis_unit == "Sentence":
    docs = _paragraphs_to_sentences(doc_paragraphs.paragraphs, min_chars=min_chars)
else:
    docs = _clean_docs(doc_paragraphs.paragraphs, min_chars=min_chars)

if len(docs) < 12:
    st.warning(
        "Not enough text after filtering. Lower 'Minimum text length' or use a longer document."
    )
    st.stop()

def _on_search_submit() -> None:
    st.session_state["_topic_active_query"] = st.session_state.get("topic_search_query", "")
    st.session_state["_topic_active_method"] = st.session_state.get("topic_search_method") or "Keyword"


@st.fragment
def _search_method_pills() -> None:
    st.session_state.setdefault("topic_search_method", "Keyword")
    st.pills(
        "Search type",
        options=["Keyword", "Regex", "Relevance", "Semantic"],
        key="topic_search_method",
        label_visibility="collapsed",
        selection_mode="single",
    )


active_query = st.session_state.get("_topic_active_query", "")
active_method = st.session_state.get("_topic_active_method", "Keyword")

rank_limit = st.session_state.get(KEY_TOPIC_RANK_LIMIT, 200)
semantic_min_score = st.session_state.get(KEY_TOPIC_SEMANTIC_MIN, 0.20)

if active_method in ("Relevance", "Semantic"):
    st.sidebar.markdown("### Search")
    rank_limit = st.sidebar.slider(
        "Result limit",
        min_value=10,
        max_value=500,
        value=200,
        step=10,
        key="topic_rank_limit",
    )

if active_method == "Semantic":
    semantic_min_score = st.sidebar.slider(
        "Minimum score",
        min_value=-1.0,
        max_value=1.0,
        value=0.20,
        step=0.01,
        key="topic_semantic_min",
        help="Higher values return stricter matches.",
    )

normalized_query = active_query.strip().lower()
coarse_default, medium_default, fine_default = _default_granularity(len(docs))
st.sidebar.markdown("### Topics (Broad to Detailed)")
st.sidebar.caption(
    "Adjust topic level granularity.\n"
    "- Higher value: fewer, broader topics.\n"
    "- Lower value: more, narrower topics."
)
coarse_size = st.sidebar.slider(
    "Broad topics",
    1,
    max(3, min(50, len(docs) // 4)),
    st.session_state.get("_topic_pref_coarse_size", coarse_default),
    key="topic_coarse_size",
    help="Start here for high-level topics. Increase to merge topics into larger groups.",
)
st.session_state["_topic_pref_coarse_size"] = coarse_size

medium_size = st.sidebar.slider(
    "Mid-level topics",
    1,
    max(3, min(30, len(docs) // 6)),
    st.session_state.get("_topic_pref_medium_size", medium_default),
    key="topic_medium_size",
    help="Balances specificity and stability. Usually between broad and detailed topic values.",
)
st.session_state["_topic_pref_medium_size"] = medium_size

fine_size = st.sidebar.slider(
    "Detailed topics",
    1,
    max(3, min(15, len(docs) // 10)),
    st.session_state.get("_topic_pref_fine_size", fine_default),
    key="topic_fine_size",
    help="Use lower values to surface niche sub-topics. Too low can produce noisy clusters.",
)
st.session_state["_topic_pref_fine_size"] = fine_size

granularity_sizes = sorted({coarse_size, medium_size, fine_size}, reverse=True)

with st.sidebar.expander("Seed words (advanced)", expanded=False):
    st.caption(
        "Guide topics by entering seed words — one group per line, words separated by commas.\n\n"
        "Example:\n```\nprivacy, personal data, consent\nliability, damages\ntermination, expiry\n```"
    )
    seed_words_raw = st.text_area(
        "Seed words",
        label_visibility="collapsed",
        key="topic_seed_words",
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

_topic_state_key = "|".join([
    hashlib.md5(file_bytes).hexdigest(),
    str(analysis_unit),
    str(min_chars),
    embedding_model_name,
    str(granularity_sizes),
    (seed_words_raw or "").strip(),
])

if st.session_state.get("_topic_state_key") != _topic_state_key:
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
            topic_model, topics = _fit_topics(
                docs_tuple, embeddings, min_topic_size, seed_topic_list
            )
            labels = _topic_labels(topic_model, topics)
            label_layers.append(labels)
            n_topics = len({t for t in topics if t != -1})
            topic_counts.append(n_topics)

    st.session_state.update({
        "_topic_state_key": _topic_state_key,
        "_topic_docs": docs,
        "_topic_embeddings": embeddings,
        "_topic_reduced": reduced_embeddings,
        "_topic_label_layers": label_layers,
        "_topic_counts": topic_counts,
    })
else:
    docs = st.session_state["_topic_docs"]
    embeddings = st.session_state["_topic_embeddings"]
    reduced_embeddings = st.session_state["_topic_reduced"]
    label_layers = st.session_state["_topic_label_layers"]
    topic_counts = st.session_state["_topic_counts"]

st.sidebar.markdown("### Topics")
sidebar_stats_cols = st.sidebar.columns(len(granularity_sizes))
for idx, size in enumerate(granularity_sizes):
    topic_name = "Broad" if idx == 0 else "Mid" if idx == 1 else "Detailed"
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
elif active_method == "Keyword":
    matched_indices = [
        idx for idx, text in enumerate(docs) if normalized_query in text.lower()
    ]
elif active_method == "Regex":
    try:
        pattern = re.compile(active_query.strip(), flags=re.IGNORECASE)
        matched_indices = [idx for idx, text in enumerate(docs) if pattern.search(text)]
    except re.error as e:
        st.warning(f"Invalid regex: {e}")
        matched_indices = []
elif active_method == "Relevance":
    scores = bm25_scores(docs, normalized_query)
    ranked = np.argsort(-scores)
    matched_indices = [int(i) for i in ranked if scores[i] > 0][:rank_limit]
    score_map = {int(i): float(scores[i]) for i in matched_indices}
else:
    encoder = get_sentence_transformer(embedding_model_name)
    query_embedding = encoder.encode([normalized_query], show_progress_bar=False)[0]
    doc_norms = np.linalg.norm(embeddings, axis=1)
    query_norm = float(np.linalg.norm(query_embedding))
    cosine_scores = (embeddings @ query_embedding) / np.clip(
        doc_norms * query_norm, 1e-9, None
    )
    ranked = np.argsort(-cosine_scores)
    matched_indices = [
        int(i) for i in ranked if cosine_scores[i] >= semantic_min_score
    ][:rank_limit]
    score_map = {int(i): float(cosine_scores[i]) for i in matched_indices}

def _show_map(
    plot_embeddings: np.ndarray,
    plot_label_layers: tuple,
    docs: tuple[str, ...],
    matched_indices: list[int],
    n_plot: int,
    zoom: float,
    html_content: str | None,
) -> None:
    expanded = n_plot >= 100
    with st.expander("Topic map", expanded=expanded):
        if html_content is None:
            st.caption("Interactive map is not available for this dataset.")
            with st.spinner("Building static map…"):
                png_b64 = _render_static_map(plot_embeddings, plot_label_layers)
            st.image(base64.b64decode(png_b64), width="stretch")
            return

        _map_options = ["Interactive", "Static"]
        st.session_state.setdefault("_topic_map_type_pref", "Interactive")
        _map_index = _map_options.index(st.session_state["_topic_map_type_pref"])
        map_type = st.radio(
            "Map type",
            _map_options,
            index=_map_index,
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state["_topic_map_type_pref"] = map_type

        if map_type == "Static":
            with st.spinner("Building static map…"):
                png_b64 = _render_static_map(plot_embeddings, plot_label_layers)
            st.image(base64.b64decode(png_b64), width="stretch")
            return

        components.html(html_content, height=860, scrolling=False)


st.subheader("Topic Explorer")

if not matched_indices:
    st.info("No text matches your search. Try a broader term.")
elif len(matched_indices) < 10:
    st.info("Too few matches to render a map (minimum 10). Results are shown in the table below.")
else:
    plot_embeddings = reduced_embeddings[matched_indices]
    plot_label_layers = tuple(labels[matched_indices] for labels in label_layers)
    n_plot = len(matched_indices)
    zoom = max(0.33, min(1.0, 15 / n_plot))

    _map_sig = (tuple(matched_indices), tuple(granularity_sizes), embedding_model_name)
    if st.session_state.get("_topic_map_sig") != _map_sig:
        with st.spinner("Building map…"):
            html_content = _render_interactive_map(
                plot_embeddings, plot_label_layers, docs, tuple(matched_indices), zoom
            )
        st.session_state["_topic_map_sig"] = _map_sig
        st.session_state["_topic_map_html"] = html_content
    else:
        html_content = st.session_state["_topic_map_html"]

    _show_map(plot_embeddings, plot_label_layers, docs, matched_indices, n_plot, zoom, html_content)

st.markdown("#### Search")
st.text_input(
    "Filter text",
    label_visibility="collapsed",
    key="topic_search_query",
    placeholder="Search across all topics…",
    on_change=_on_search_submit,
)
_search_method_pills()

max_rows = st.sidebar.slider(
    "Max rows", min_value=10, max_value=500, value=100, step=10, key="topic_max_rows"
)


@st.fragment
def _results_section(
    docs: tuple[str, ...],
    label_layers: list[np.ndarray],
    matched_indices: list[int],
    score_map: dict[int, float],
    search_query: str,
    search_method: str,
    max_rows: int,
    granularity_sizes: list[int],
) -> None:
    topic_columns = ["Broad topics", "Mid-level topics", "Detailed topics"]
    results_df = pd.DataFrame(
        {
            "paragraph_idx": np.arange(len(docs)),
            "text": docs,
        }
    )
    for idx, size in enumerate(granularity_sizes):
        topic_col_name = (
            topic_columns[idx] if idx < len(topic_columns) else f"Topics {idx + 1}"
        )
        results_df[topic_col_name] = label_layers[idx]

    results_df = (
        results_df.iloc[matched_indices].reset_index(drop=True)
        if matched_indices
        else results_df.iloc[0:0]
    )
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
            key="topic_filter_col",
            label_visibility="collapsed",
        )
        topic_options = sorted(t for t in results_df[filter_col].unique() if t != "Noise")
        selected_topics = fcol2.multiselect(
            "Topics",
            options=topic_options,
            default=[],
            key="topic_filter_values",
            placeholder=f"Select {filter_col.lower()}…",
            label_visibility="collapsed",
        )
        if selected_topics:
            results_df = results_df[
                results_df[filter_col].isin(selected_topics)
            ].reset_index(drop=True)

    st.markdown("#### Results")
    sortable_cols = [c for c in ["score", *topic_columns, "paragraph_idx"] if c in results_df.columns]
    has_score = bool(score_map)
    if has_score != st.session_state.get("_topic_had_score"):
        st.session_state["_topic_had_score"] = has_score
        st.session_state["topic_sort_col"] = "score" if has_score else "paragraph_idx"
        st.session_state["topic_sort_asc"] = not has_score
    scol1, scol2 = st.columns([3, 1])
    sort_by = scol1.selectbox(
        "Sort by",
        options=sortable_cols,
        index=0,
        key="topic_sort_col",
        label_visibility="collapsed",
    )
    sort_asc = scol2.toggle("Ascending", value=False, key="topic_sort_asc")
    results_df = results_df.sort_values(sort_by, ascending=sort_asc).reset_index(drop=True)

    st.caption(f"{len(results_df)} passages")
    st.dataframe(
        results_df.head(max_rows),
        width="stretch",
        hide_index=True,
        column_config={
            "paragraph_idx": st.column_config.NumberColumn("Para", width="small"),
            "text": st.column_config.TextColumn("Passage", width="large"),
            "score": st.column_config.NumberColumn("Score", format="%.4f", width="small"),
        },
    )

    show_markdown = st.checkbox(
        "Expand results",
        key="topic_print_markdown",
        value=False,
    )

    if not show_markdown:
        return

    shown_df = results_df.head(max_rows).copy()
    if shown_df.empty:
        st.markdown("_No matching rows._")
        return

    topic_cols = [col for col in topic_columns if col in shown_df.columns]
    finest_labels = label_layers[-1]
    color_map = _topic_color_map(finest_labels)
    lines: list[str] = []
    for _, row in shown_df.iterrows():
        para_idx = int(row["paragraph_idx"])
        text = str(row["text"])
        if search_method in ("Relevance", "Semantic"):
            highlighted = _highlight_query_tokens(text, search_query)
        elif search_method == "Regex":
            try:
                highlighted = re.compile(
                    search_query.strip(), flags=re.IGNORECASE
                ).sub(lambda m: f"<mark>{m.group(0)}</mark>", text)
            except re.error:
                highlighted = text
        else:
            highlighted = _highlight_term(text, search_query)
        topic_label = finest_labels[para_idx] if para_idx < len(finest_labels) else ""
        color = color_map.get(str(topic_label), "#ffe066")
        highlighted = _highlight_topic_keywords(highlighted, str(topic_label), color)
        topics = " → ".join(f"{row[col]}" for col in topic_cols)
        lines.append(f"#### Para {para_idx} — {topics}\n\n{highlighted}")
    st.markdown("\n\n---\n\n".join(lines), unsafe_allow_html=True)


_results_section(
    docs=docs,
    label_layers=label_layers,
    matched_indices=matched_indices,
    score_map=score_map,
    search_query=active_query,
    search_method=active_method,
    max_rows=max_rows,
    granularity_sizes=granularity_sizes,
)
