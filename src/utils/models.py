"""
Cached model loaders shared across pages.
"""

import spacy
import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer


@st.cache_resource(show_spinner=False)
def get_sentence_transformer(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def get_cross_encoder(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


@st.cache_resource(show_spinner=False)
def get_spacy_nlp(model_name: str = "en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Run: python -m spacy download {model_name}"
        ) from None
