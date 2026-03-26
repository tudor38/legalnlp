"""
Shared text processing utilities.
"""

import math
import re
from collections.abc import Sequence

import numpy as np
import Stemmer as _PyStemmer
from spacy.lang.en.stop_words import STOP_WORDS

from src.stats.config import CFG

_stemmer = _PyStemmer.Stemmer("english")

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


def highlight_term(text: str, query: str, color: str = "") -> str:
    if not query.strip():
        return text
    style = f' style="background:{color}"' if color else ""
    return re.compile(re.escape(query), flags=re.IGNORECASE).sub(
        lambda m: f"<mark{style}>{m.group(0)}</mark>", text
    )


def highlight_query_tokens(text: str, query: str, color: str = "") -> str:
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


def highlight_topic_keywords(text: str, topic_label: str, color: str) -> str:
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


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def bm25_scores(
    docs: Sequence[str],
    query: str,
    k1: float = CFG.search.bm25_k1,
    b: float = CFG.search.bm25_b,
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
