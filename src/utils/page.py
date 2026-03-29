"""
Page-level helpers shared across Streamlit pages.
"""

import streamlit as st

from src.app_state import get_file_bytes, make_store


def expanded_view_controls(
    expanded_key: str,
    collapse_key: str,
) -> tuple[bool, bool]:
    """Render 'Show expanded view' and 'Collapse all' toggles.

    Uses permanent session-state keys for navigation-safe persistence.
    Widget keys are derived by prepending '_' to each permanent key.

    Returns:
        (expanded, collapse) — both False when expanded view is off.
    """
    col_v, col_c = st.columns([1, 1])
    expanded = col_v.toggle(
        "Show expanded view",
        value=st.session_state.get(expanded_key, False),
        key=f"_{expanded_key}",
        on_change=make_store(expanded_key),
    )
    if expanded:
        collapse = col_c.toggle(
            "Collapse all",
            value=st.session_state.get(collapse_key, False),
            key=f"_{collapse_key}",
            on_change=make_store(collapse_key),
        )
    else:
        collapse = False
    return expanded, collapse


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
