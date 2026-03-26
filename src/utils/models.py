"""
Cached model loaders shared across pages.
"""

import contextlib
import io
import logging

import spacy
import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer

# Suppress the "LOAD REPORT" stdout noise and transformer info logs that
# sentence-transformers 5.x prints on every model load.
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


@contextlib.contextmanager
def _quiet():
    """Redirect stdout to swallow progress-bar / load-report prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


@st.cache_resource(show_spinner=False, max_entries=4)
def get_sentence_transformer(model_name: str) -> SentenceTransformer:
    with _quiet():
        return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False, max_entries=4)
def get_cross_encoder(model_name: str) -> CrossEncoder:
    with _quiet():
        return CrossEncoder(model_name)


@st.cache_resource(show_spinner=False, max_entries=4)
def get_spacy_nlp(model_name: str = "en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed."
        ) from None
    except ValueError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is missing a required component: {exc}"
        ) from None
