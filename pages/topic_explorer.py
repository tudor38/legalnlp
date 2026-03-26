import base64
import hashlib
import io
import re
from collections.abc import Sequence

import datamapplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from src.app_state import MODEL_MINILM, MODEL_MPNET
from src.comments.extract import extract_paragraphs
from src.utils.models import get_sentence_transformer
from src.utils.page import require_document
from src.utils.text import (
    TOPIC_PALETTE,
    bm25_scores,
    highlight_query_tokens,
    highlight_term,
    highlight_topic_keywords,
    tokenize,
)


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


@st.cache_resource(show_spinner=False, max_entries=3)
def _embed_docs(docs: tuple[str, ...], embedding_model_name: str) -> np.ndarray:
    encoder = get_sentence_transformer(embedding_model_name)
    return encoder.encode(list(docs), show_progress_bar=False)


@st.cache_resource(show_spinner=False, max_entries=9)
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


@st.cache_resource(show_spinner=False, max_entries=3)
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


@st.cache_data(show_spinner=False, max_entries=5)
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


@st.cache_data(show_spinner=False, max_entries=5)
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
            enable_search=False,
            noise_label="Noise",
            noise_color="#cccccc",
            initial_zoom_fraction=zoom,
        )
        return str(plot)
    except (IndexError, ValueError):
        return None


def _default_granularity(n_docs: int) -> tuple[int, int, int]:
    highlevel = max(4, n_docs // 4)
    midlevel = max(3, n_docs // 10)
    lowlevel = 2
    # Enforce strictly decreasing so all three levels stay distinct after set deduplication
    midlevel = min(midlevel, highlevel - 1)
    if midlevel <= lowlevel:
        midlevel = lowlevel + 1
    if highlevel <= midlevel:
        highlevel = midlevel + 1
    return highlevel, midlevel, lowlevel


def _topic_color_map(label_layer: np.ndarray) -> dict[str, str]:
    unique = [l for l in dict.fromkeys(label_layer) if l != "Noise"]
    return {
        label: TOPIC_PALETTE[i % len(TOPIC_PALETTE)] for i, label in enumerate(unique)
    }


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
_selected_model = st.sidebar.selectbox(
    "Embedding model",
    options=_model_options + ["Custom…"],
    index=_model_options.index(_saved_model) if _saved_model in _model_options else 0,
    key="topic_embedding_model",
    help="This model converts each paragraph into vectors before topic clustering.",
)
if _selected_model == "Custom…":
    embedding_model_name = st.sidebar.text_input(
        "HuggingFace model ID",
        value=st.session_state.get("_topic_pref_custom_model", ""),
        placeholder="e.g. BAAI/bge-small-en-v1.5",
        key="topic_custom_model",
    ).strip()
    st.session_state["_topic_pref_custom_model"] = embedding_model_name
    if not embedding_model_name:
        st.info(
            "Enter a HuggingFace model ID in the sidebar to use a custom embedding model."
        )
        st.stop()
else:
    embedding_model_name = _selected_model
st.session_state["_topic_pref_embedding_model"] = embedding_model_name

if analysis_unit == "Sentence":
    docs = _paragraphs_to_sentences(doc_paragraphs.paragraphs, min_chars=min_chars)
else:
    docs = _clean_docs(doc_paragraphs.paragraphs, min_chars=min_chars)

if len(docs) < 12:
    st.warning(
        f"Only {len(docs)} passage{'s' if len(docs) != 1 else ''} remain after filtering "
        f"(minimum 12 required). "
        f"Try lowering **Minimum text length** (currently {min_chars} chars) "
        f"or switching the analysis unit to **Sentence**."
    )
    st.stop()

st.sidebar.metric("Passages", len(docs))


def _on_search_submit() -> None:
    st.session_state["_topic_active_query"] = st.session_state.get(
        "topic_search_query", ""
    )
    st.session_state["_topic_active_method"] = (
        st.session_state.get("topic_search_method") or "Keyword"
    )


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
highlevel_default, midlevel_default, lowlevel_default = _default_granularity(len(docs))
_highlevel_max = max(3, min(50, len(docs) // 4))
_midlevel_max = max(3, min(30, len(docs) // 6))
_lowlevel_max = max(3, min(15, len(docs) // 10))


def _to_slider(min_topic_size: int, max_val: int) -> int:
    """Convert min_topic_size → normalized 1–100 slider position."""
    if max_val <= 2:
        return 1
    return max(1, min(100, round((max_val - min_topic_size) * 99 / (max_val - 2)) + 1))


def _from_slider(pos: int, max_val: int) -> int:
    """Convert normalized 1–100 slider position → min_topic_size."""
    if max_val <= 2:
        return 2
    return max(2, round(max_val - (pos - 1) * (max_val - 2) / 99))


st.sidebar.markdown("### Topic Levels")

_highlevel_pos = st.sidebar.slider(
    "High-level",
    1,
    100,
    st.session_state.get(
        "_topic_pref_highlevel", _to_slider(highlevel_default, _highlevel_max)
    ),
    key="topic_highlevel",
    help="Move right for more high-level topics; move left for fewer, broader groupings.",
)
st.session_state["_topic_pref_highlevel"] = _highlevel_pos
highlevel_size = _from_slider(_highlevel_pos, _highlevel_max)

_midlevel_pos = st.sidebar.slider(
    "Mid-level",
    1,
    100,
    st.session_state.get(
        "_topic_pref_midlevel", _to_slider(midlevel_default, _midlevel_max)
    ),
    key="topic_midlevel",
    help="Move right for more mid-level topics; move left for fewer, broader groupings.",
)
st.session_state["_topic_pref_midlevel"] = _midlevel_pos
midlevel_size = _from_slider(_midlevel_pos, _midlevel_max)

_lowlevel_pos = st.sidebar.slider(
    "Low-level",
    1,
    100,
    st.session_state.get(
        "_topic_pref_lowlevel", _to_slider(lowlevel_default, _lowlevel_max)
    ),
    key="topic_lowlevel",
    help="Move right for more low-level topics; move left for fewer, broader groupings.",
)
st.session_state["_topic_pref_lowlevel"] = _lowlevel_pos
lowlevel_size = _from_slider(_lowlevel_pos, _lowlevel_max)

granularity_sizes = [highlevel_size, midlevel_size, lowlevel_size]

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
        tuple(w.strip() for w in line.split(",") if len(w.strip()) > 1)
        for line in seed_words_raw.strip().splitlines()
        if line.strip()
    )
    parsed = tuple(group for group in parsed if group)
    if parsed:
        seed_topic_list = parsed
    else:
        st.sidebar.warning(
            "Seed words ignored — no valid words found (each word must be at least 2 characters)."
        )

_topic_state_key = hashlib.md5(
    repr(
        {
            "doc": hashlib.md5(file_bytes).hexdigest(),
            "unit": analysis_unit,
            "min_chars": min_chars,
            "model": embedding_model_name,
            "granularity": granularity_sizes,
            "seeds": (seed_words_raw or "").strip(),
        }
    ).encode()
).hexdigest()

if st.session_state.get("_topic_state_key") != _topic_state_key:
    try:
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

    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    st.session_state.update(
        {
            "_topic_state_key": _topic_state_key,
            "_topic_docs": docs,
            "_topic_embeddings": embeddings,
            "_topic_reduced": reduced_embeddings,
            "_topic_label_layers": label_layers,
            "_topic_counts": topic_counts,
        }
    )
else:
    docs = st.session_state["_topic_docs"]
    embeddings = st.session_state["_topic_embeddings"]
    reduced_embeddings = st.session_state["_topic_reduced"]
    label_layers = st.session_state["_topic_label_layers"]
    topic_counts = st.session_state["_topic_counts"]

st.sidebar.markdown("### Topics Found")
sidebar_stats_cols = st.sidebar.columns(len(granularity_sizes))
for idx, size in enumerate(granularity_sizes):
    topic_name = "High-level" if idx == 0 else "Mid-level" if idx == 1 else "Low-level"
    sidebar_stats_cols[idx].metric(topic_name, topic_counts[idx])

has_noise = any((labels == "Noise").any() for labels in label_layers)
if has_noise:
    st.sidebar.caption(
        "Noise: text that doesn't cluster into any topic. "
        "Raise minimum text length or adjust topic levels to reduce noise."
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
    zoom: float,
    html_content: str | None,
) -> None:
    if html_content is None:
        st.caption(
            "Interactive map is not available for this dataset. Adjusting topic sliders could enable the option to view an interactive map."
        )
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
    st.info(
        "Too few matches to render a map (minimum 10). Results are shown in the table below."
    )
else:
    plot_embeddings = reduced_embeddings[matched_indices]
    plot_label_layers = tuple(labels[matched_indices] for labels in label_layers)
    zoom = max(0.33, min(1.0, 15 / len(matched_indices)))

    _map_sig = (tuple(matched_indices), tuple(granularity_sizes), embedding_model_name)
    if st.session_state.get("_topic_map_sig") != _map_sig:
        _n_unique = len({lbl for lbl in plot_label_layers[-1] if lbl != "Noise"})
        st.session_state["_topic_map_type_pref"] = (
            "Static" if _n_unique <= 6 else "Interactive"
        )
        st.session_state["_topic_map_sig"] = _map_sig
        with st.spinner("Building map…"):
            html_content = _render_interactive_map(
                plot_embeddings, plot_label_layers, docs, tuple(matched_indices), zoom
            )
    else:
        html_content = _render_interactive_map(
            plot_embeddings, plot_label_layers, docs, tuple(matched_indices), zoom
        )

    _show_map(plot_embeddings, plot_label_layers, zoom, html_content)

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
    topic_columns = ["High-level topics", "Mid-level topics", "Low-level topics"]
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
        results_df = results_df.sort_values("score", ascending=False).reset_index(
            drop=True
        )

    selected_topics: list[str] = []

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
        topic_options = sorted(
            t for t in results_df[filter_col].unique() if t != "Noise"
        )
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
    sortable_cols = [
        c for c in ["score", *topic_columns, "paragraph_idx"] if c in results_df.columns
    ]
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
    results_df = results_df.sort_values(sort_by, ascending=sort_asc).reset_index(
        drop=True
    )

    st.caption(f"{len(results_df)} results")
    st.dataframe(
        results_df.head(max_rows),
        width="stretch",
        hide_index=True,
        column_config={
            "paragraph_idx": st.column_config.NumberColumn("Para", width="small"),
            "text": st.column_config.TextColumn("Passage", width="large"),
            "score": st.column_config.NumberColumn(
                "Score", format="%.4f", width="small"
            ),
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
            highlighted = highlight_query_tokens(text, search_query)
        elif search_method == "Regex":
            try:
                highlighted = re.compile(search_query.strip(), flags=re.IGNORECASE).sub(
                    lambda m: f"<mark>{m.group(0)}</mark>", text
                )
            except re.error:
                highlighted = text
        else:
            highlighted = highlight_term(text, search_query)
        topic_label = finest_labels[para_idx] if para_idx < len(finest_labels) else ""
        color = color_map.get(str(topic_label), "#ffe066")
        highlighted = highlight_topic_keywords(highlighted, str(topic_label), color)
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
