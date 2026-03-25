import io

import streamlit as st

from src.app_state import KEY_GRAPH_DATA
from src.comments.extract import extract_paragraphs
from src.nlp.graph import (
    ALL_NODE_TYPES,
    ALL_RELATION_TYPES,
    NODE_COLORS,
    build_figure,
    build_graph,
    deserialize_graph,
    edges_to_dataframe,
)
from src.utils.page import require_document

# ---------------------------------------------------------------------------
# Page guard + document loading
# ---------------------------------------------------------------------------

file_bytes = require_document()
doc_paragraphs = extract_paragraphs(io.BytesIO(file_bytes))
paragraphs = tuple(p.strip() for p in doc_paragraphs.paragraphs if p and p.strip())

if not paragraphs:
    st.warning("No usable text found in the document.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Filters")

parties_only = st.sidebar.toggle(
    "Parties only",
    value=True,
    help="Show only ORG/PERSON nodes and edges between them.",
)

entity_types = st.sidebar.multiselect(
    "Node types",
    options=ALL_NODE_TYPES,
    default=["PARTY", "GPE"],
    disabled=parties_only,
    help="Ignored when 'Parties only' is on.",
)

relation_types = st.sidebar.multiselect(
    "Relationship types",
    options=ALL_RELATION_TYPES,
    default=ALL_RELATION_TYPES,
)

min_confidence = st.sidebar.slider(
    "Min. confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.05,
)

show_labels = st.sidebar.toggle("Show edge labels", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model")
model_name = st.sidebar.selectbox(
    "spaCy model",
    options=["en_core_web_sm", "en_core_web_trf"],
    index=0,
    help=(
        "**sm** — fast, good for most contracts.\n\n"
        "**trf** — transformer-based, ~93%+ NER accuracy. CPU-compatible but slower. "
        "Install via: `uv pip install \"en_core_web_trf @ <wheel-url>\"` "
        "(click Build Graph for the exact command if not installed)."
    ),
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.subheader("Relationship Graph")
st.markdown(
    "Who are the parties, what did they agree to, and who does what to whom? "
    "Click a node to inspect it. Review the table below to verify."
)

# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

col_btn, col_clear = st.columns([2, 1])
with col_btn:
    build_clicked = st.button("Build Graph", type="primary", width="stretch")
with col_clear:
    if st.button("Clear", width="stretch"):
        st.session_state.pop(KEY_GRAPH_DATA, None)
        st.rerun()

if build_clicked:
    st.session_state.pop(KEY_GRAPH_DATA, None)
    try:
        with st.spinner("Extracting entities and relationships…"):
            graph_dict, meta = build_graph(paragraphs, model_name)
        st.session_state[KEY_GRAPH_DATA] = (graph_dict, meta)
    except RuntimeError as exc:
        if "not installed" in str(exc) or "missing a required component" in str(exc):
            import spacy as _spacy
            v = _spacy.__version__
            major_minor = ".".join(v.split(".")[:2])
            model_ver = f"{major_minor}.0"
            wheel = (
                f"https://github.com/explosion/spacy-models/releases/download/"
                f"{model_name}-{model_ver}/{model_name}-{model_ver}-py3-none-any.whl"
            )
            extra = (
                " spacy-curated-transformers" if model_name == "en_core_web_trf" else ""
            )
            st.error(
                f"**spaCy model `{model_name}` could not be loaded.**\n\n"
                f"Install it with uv (direct wheel, bypasses project resolution):\n"
                f"```\nuv pip install \"{model_name} @ {wheel}\"{extra}\n```"
            )
        else:
            raise

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

if KEY_GRAPH_DATA not in st.session_state:
    st.stop()

graph_dict, meta = st.session_state[KEY_GRAPH_DATA]
G = deserialize_graph(graph_dict)

# Stat metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Nodes", meta["n_nodes"])
m2.metric("Edges", meta["n_edges"])
m3.metric("Parties", meta["n_parties"])
m4.metric("Relation types", len(meta["relation_types"]))

st.markdown("")

# Graph figure
selected_types_set = set(entity_types) if not parties_only else {"PARTY"}
selected_rels_set = set(relation_types) if relation_types else set(ALL_RELATION_TYPES)

fig = build_figure(
    G,
    selected_types=selected_types_set,
    selected_relations=selected_rels_set,
    min_confidence=min_confidence,
    parties_only=parties_only,
    show_labels=show_labels,
)

event = st.plotly_chart(
    fig,
    width="stretch",
    on_select="rerun",
    selection_mode="points",
    key="rg_chart",
)

# ---------------------------------------------------------------------------
# Selected node detail
# ---------------------------------------------------------------------------

selected_node: str | None = None
if event:
    points = (event.get("selection") or {}).get("points") or []
    if points:
        selected_node = points[0].get("customdata")

if selected_node and selected_node in G.nodes:
    nd = G.nodes[selected_node]
    with st.expander(f"Node: **{selected_node}**", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            ntype = nd.get("type", "?")
            color = NODE_COLORS.get(ntype, "#999")
            st.markdown(
                f'**Type:** <span style="color:{color};font-weight:bold">{ntype}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Mentions:** {nd.get('mention_count', '?')}")
            paras = nd.get("para_indices", [])
            para_str = ", ".join(f"¶{i}" for i in sorted(paras)[:6])
            if len(paras) > 6:
                para_str += f" +{len(paras) - 6} more"
            st.markdown(f"**Paragraphs:** {para_str}")
        with c2:
            aliases = nd.get("aliases", [])
            if aliases:
                st.markdown("**Surface forms:**")
                for a in aliases[:8]:
                    st.markdown(f"- {a}")

        ctx = nd.get("first_context", "")
        if ctx:
            st.markdown("**First occurrence:**")
            st.markdown(f"> {ctx[:300]}")

# ---------------------------------------------------------------------------
# Edge table
# ---------------------------------------------------------------------------

st.markdown("#### Extracted Relationships")

edge_df = edges_to_dataframe(G, min_confidence=min_confidence)

if not edge_df.empty and selected_rels_set != set(ALL_RELATION_TYPES):
    edge_df = edge_df[edge_df["Relation"].isin(selected_rels_set)]
if not edge_df.empty and parties_only:
    party_nodes = {n for n, d in G.nodes(data=True) if d.get("type") == "PARTY"}
    edge_df = edge_df[edge_df["From"].isin(party_nodes) & edge_df["To"].isin(party_nodes)]

if edge_df.empty:
    st.info("No relationships match the current filters.")
else:
    st.caption(f"{len(edge_df)} relationship{'s' if len(edge_df) != 1 else ''}")
    st.dataframe(
        edge_df,
        width="stretch",
        hide_index=True,
        column_config={
            "From": st.column_config.TextColumn("From", width="medium"),
            "Relation": st.column_config.TextColumn("Relation", width="medium"),
            "To": st.column_config.TextColumn("To", width="medium"),
            "Verb": st.column_config.TextColumn("Verb", width="small"),
            "¶": st.column_config.NumberColumn("¶", width="small"),
            "Date": st.column_config.TextColumn("Date", width="small"),
            "Amount": st.column_config.TextColumn("Amount", width="small"),
            "Conf.": st.column_config.NumberColumn("Conf.", width="small", format="%.2f"),
            "Method": st.column_config.TextColumn("Method", width="small"),
            "Sentence": st.column_config.TextColumn("Sentence", width="large"),
        },
    )
