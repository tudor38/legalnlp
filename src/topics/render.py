"""
Topic map rendering: static PNG, interactive HTML, and the map UI widget.
"""

import base64
import io

import datamapplot
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from src.app_state import KEY_TOPIC_MAP_TYPE_PREF


@st.cache_data(show_spinner=False, max_entries=5)
def render_static_map(
    plot_embeddings: np.ndarray,
    plot_label_layers: tuple,
) -> str:
    """Return base64-encoded PNG of the topic map."""
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
def render_interactive_map(
    plot_embeddings: np.ndarray,
    plot_label_layers: tuple,
    all_docs: tuple[str, ...],
    matched_indices: tuple[int, ...],
    zoom: float,
) -> str | None:
    """Return HTML string, or None if datamapplot cannot render interactively.

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


def show_map(
    plot_embeddings: np.ndarray,
    plot_label_layers: tuple,
    zoom: float,
    html_content: str | None,
) -> None:
    """Render the topic map widget: Interactive or Static, with a type selector."""
    if html_content is None:
        st.caption(
            "Interactive map is not available for this dataset. "
            "Adjusting topic sliders could enable the option to view an interactive map."
        )
        with st.spinner("Building static map…"):
            png_b64 = render_static_map(plot_embeddings, plot_label_layers)
        st.image(base64.b64decode(png_b64), width="stretch")
        return

    _map_options = ["Interactive", "Static"]
    st.session_state.setdefault(KEY_TOPIC_MAP_TYPE_PREF, "Interactive")
    _map_index = _map_options.index(st.session_state[KEY_TOPIC_MAP_TYPE_PREF])
    map_type = st.radio(
        "Map type",
        _map_options,
        index=_map_index,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state[KEY_TOPIC_MAP_TYPE_PREF] = map_type

    if map_type == "Static":
        with st.spinner("Building static map…"):
            png_b64 = render_static_map(plot_embeddings, plot_label_layers)
        st.image(base64.b64decode(png_b64), width="stretch")
        return

    components.html(html_content, height=860, scrolling=False)
