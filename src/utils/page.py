"""
Page-level helpers shared across Streamlit pages.
"""

import streamlit as st

from src.app_state import get_file_bytes


def require_document() -> bytes:
    """
    Return the uploaded document bytes from session state, or stop the page
    with a clear navigation hint if no document has been loaded yet.

    All analysis pages call this at the top to enforce a single upload point.
    """
    file_bytes = get_file_bytes()
    if not file_bytes:
        st.info(
            "No document loaded. "
            "Upload a Word document on the **Document Statistics** page first."
        )
        st.stop()
    return file_bytes
