import io

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import CrossEncoder

from src.comments.extract import extract_paragraphs

# label order for cross-encoder NLI models (MNLI convention)
_NLI_LABELS = ["Contradicting", "Supporting", "Neutral"]
_LABEL_SORT_ORDER = {"Supporting": 0, "Contradicting": 1, "Neutral": 2}
_LABEL_DISPLAY = {
    "Supporting": '<span style="color:#2e7d32;font-weight:600">✓ Supporting</span>',
    "Contradicting": '<span style="color:#c62828;font-weight:600">✗ Contradicting</span>',
}



@st.cache_resource(show_spinner=False)
def _load_model(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


@st.cache_data(show_spinner=False)
def _score_passages(
    _model: CrossEncoder,
    model_name: str,
    hypothesis: str,
    docs: tuple[str, ...],
) -> np.ndarray:
    """Returns array of shape (n_docs, 3): [contradiction, entailment, neutral]."""
    pairs = [(doc, hypothesis) for doc in docs]
    return _model.predict(pairs)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
file_bytes = st.session_state.get("p1_file_bytes")
if not file_bytes:
    st.info("Upload a document on Page 1 to get started.")
    st.stop()

doc_paragraphs = extract_paragraphs(io.BytesIO(file_bytes))
docs = tuple(p.strip() for p in doc_paragraphs.paragraphs if p and len(p.strip()) >= 40)

if not docs:
    st.warning("No usable text found in the document.")
    st.stop()

# Sidebar
st.sidebar.markdown("### Model")
model_name = st.sidebar.selectbox(
    "NLI model",
    options=["cross-encoder/nli-deberta-v3-base", "cross-encoder/nli-deberta-v3-small",],
    index=0,
    help="Larger model is more accurate but slower.",
)
min_confidence = st.sidebar.slider(
    "Minimum confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.01,
    help="Minimum confidence of the predicted label to include a passage.",
)

# Main
st.subheader("Find Basis")

hypothesis = st.text_area(
    "State your claim",
    key="p2_custom",
    label_visibility="collapsed",
    placeholder="State a proposition, e.g. 'The Receiving Party can keep some confidential data after the contract ends.'",
    height=100,
).strip()

st.caption(
    "Returns <span style='color:#2e7d32;font-weight:600'>✓ supporting</span> and "
    "<span style='color:#c62828;font-weight:600'>✗ contradicting</span> passages from the document. No generated content.",
    unsafe_allow_html=True,
)

if not hypothesis:
    st.stop()


with st.spinner("Running entailment analysis..."):
    model = _load_model(model_name)
    raw_scores = _score_passages(model, model_name, hypothesis, docs)

# softmax to convert logits → probabilities
exp = np.exp(raw_scores - raw_scores.max(axis=1, keepdims=True))
scores = exp / exp.sum(axis=1, keepdims=True)

# scores[:, 0] = entailment, [:, 1] = contradiction, [:, 2] = neutral
label_indices = np.argmax(scores, axis=1)
confidence = scores[np.arange(len(scores)), label_indices]
labels = [_NLI_LABELS[i] for i in label_indices]

results_df = pd.DataFrame({
    "idx": np.arange(len(docs)),
    "passage": list(docs),
    "label": labels,
    "score": confidence,
})

results_df = results_df[results_df["label"] != "Neutral"]

results_df = results_df[results_df["score"] >= min_confidence]
results_df["_sort"] = results_df["label"].map(_LABEL_SORT_ORDER)
results_df = (
    results_df
    .sort_values(["_sort", "score"], ascending=[True, False])
    .drop(columns="_sort")
    .reset_index(drop=True)
)

n_supporting = (results_df["label"] == "Supporting").sum()
n_contradicting = (results_df["label"] == "Contradicting").sum()
st.caption(f"{n_supporting} supporting, {n_contradicting} contradicting")
st.dataframe(
    results_df,
    width="stretch",
    hide_index=True,
    column_config={
        "idx": st.column_config.NumberColumn("#", width="small"),
        "passage": st.column_config.TextColumn("Passage", width="large"),
        "label": st.column_config.TextColumn("Label", width="small"),
        "score": st.column_config.NumberColumn("Confidence", format="%.3f", width="small"),
    },
)

if st.checkbox("Show expanded view", key="p2_print_markdown", value=False):
    if results_df.empty:
        st.markdown("_No matching passages._")
    else:
        lines: list[str] = []
        for _, row in results_df.iterrows():
            lines.append(
                f"#### ¶{int(row['idx'])} — {_LABEL_DISPLAY.get(row['label'], row['label'])} ({row['score']:.3f})\n\n"
                f"{row['passage']}"
            )
        st.markdown("\n\n---\n\n".join(lines), unsafe_allow_html=True)
