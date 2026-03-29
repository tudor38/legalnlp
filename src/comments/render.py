import html
import streamlit as st
from annotated_text import annotated_text


def _highlight_color() -> str:
    """Return a highlight color suited to the active Streamlit theme."""
    if st.get_option("theme.base") == "dark":
        return "#B45309"
    return "#FDE68A"


def render_paragraph_with_redline_pair(para: str, deleted: str, inserted: str) -> None:
    events = []
    if deleted:
        idx = para.find(deleted)
        if idx != -1:
            events.append((idx, idx + len(deleted), "del", deleted))
    if inserted:
        idx = para.find(inserted)
        if idx != -1:
            events.append((idx, idx + len(inserted), "ins", inserted))
    if not events:
        st.markdown(f"> {para}")
        return
    events.sort(key=lambda e: e[0])
    parts = []
    pos = 0
    for start, end, kind, text in events:
        if start < pos:
            continue
        parts.append(html.escape(para[pos:start]))
        if kind == "del":
            parts.append(
                f'<span style="color:#ef4444;text-decoration:line-through">{html.escape(text)}</span>'
            )
        else:
            parts.append(
                f'<span style="color:#3b82f6;text-decoration:underline">{html.escape(text)}</span>'
            )
        pos = end
    parts.append(html.escape(para[pos:]))
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_paragraph_with_redline(para: str, text: str, kind: str) -> None:
    idx = para.find(text)
    if idx == -1:
        st.markdown(f"> {para}")
        return
    before = html.escape(para[:idx])
    after = html.escape(para[idx + len(text) :])
    escaped = html.escape(text)
    if kind == "insertion":
        styled = (
            f'<span style="color:#3b82f6;text-decoration:underline">{escaped}</span>'
        )
    else:
        styled = (
            f'<span style="color:#ef4444;text-decoration:line-through">{escaped}</span>'
        )
    st.markdown(f"{before}{styled}{after}", unsafe_allow_html=True)


def render_paragraph_with_highlight(para: str, selected: str) -> None:
    idx = para.find(selected)
    if idx == -1:
        st.markdown(f"> {para}")
        return
    before = para[:idx]
    after = para[idx + len(selected) :]
    annotated_text(before, (selected, "", _highlight_color()), after)
