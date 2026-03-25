"""
Shared text processing utilities.
"""

import math
import re
from collections.abc import Sequence

import numpy as np

TOPIC_PALETTE = [
    "#ffe066",
    "#b5ead7",
    "#b5d5ff",
    "#e8b5ff",
    "#ffb5b5",
    "#ffd9b5",
    "#b5ffe4",
    "#c9ffb5",
    "#ffb5e8",
    "#b5f0ff",
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def bm25_scores(
    docs: Sequence[str], query: str, k1: float = 1.5, b: float = 0.75
) -> np.ndarray:
    query_terms = tokenize(query)
    if not query_terms:
        return np.zeros(len(docs), dtype=float)

    tokenized_docs = [tokenize(doc) for doc in docs]
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
    max_score = scores.max()
    if max_score > 0:
        scores /= max_score
    return scores
