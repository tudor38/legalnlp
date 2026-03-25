"""
Shared NLI (Natural Language Inference) utilities.

All functions operate on CrossEncoder models following the MNLI label convention:
  index 0 → Contradicting
  index 1 → Supporting
  index 2 → Neutral
"""

import numpy as np
import streamlit as st
from sentence_transformers import CrossEncoder

# Human-readable labels in MNLI index order
NLI_LABELS = ["Contradicting", "Supporting", "Neutral"]

LABEL_SORT_ORDER = {"Supporting": 0, "Contradicting": 1, "Neutral": 2}

LABEL_DISPLAY = {
    "Supporting": '<span style="color:#2e7d32;font-weight:600">✓ Supporting</span>',
    "Contradicting": '<span style="color:#c62828;font-weight:600">✗ Contradicting</span>',
}


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


@st.cache_data(show_spinner=False)
def score_passages(
    _model: CrossEncoder,
    model_name: str,
    hypothesis: str,
    docs: tuple[str, ...],
) -> np.ndarray:
    """
    Score each document in *docs* against *hypothesis*.

    Returns a probability array of shape (n_docs, 3):
      [:, 0] Contradicting  [:, 1] Supporting  [:, 2] Neutral
    """
    pairs = [(doc, hypothesis) for doc in docs]
    raw = _model.predict(pairs)
    return softmax(raw)


def template_score(nli: CrossEncoder, premise: str, templates: list[str]) -> float:
    """Average Supporting probability of *premise* against each template."""
    if not premise.strip():
        return 0.0
    probs = softmax(nli.predict([(premise, t) for t in templates]))
    return float(probs[:, 1].mean())  # index 1 = Supporting


def contentiousness_score(nli: CrossEncoder, texts: list[str]) -> float:
    """
    Average Contradicting probability across pairs of *texts*.

    Caps at 5 texts to keep inference fast.
    Returns 0.0 when fewer than 2 non-empty texts are provided.
    """
    texts = [t.strip() for t in texts if t.strip()]
    if len(texts) < 2:
        return 0.0
    cap = min(len(texts), 5)
    pairs = [(texts[i], texts[j]) for i in range(cap) for j in range(cap) if i != j]
    probs = softmax(nli.predict(pairs))
    return float(probs[:, 0].mean())  # index 0 = Contradicting
