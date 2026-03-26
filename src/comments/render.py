import html
import re
import streamlit as st
import pandas as pd
from annotated_text import annotated_text
from datetime import datetime
from src.comments.extract import Comment, WordVersion


def _format_date(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.rstrip("Z"))
        return dt.strftime("%B %-d, %Y · %-I:%M %p")
    except ValueError:
        return iso


def _highlight_color() -> str:
    """Return a highlight color suited to the active Streamlit theme."""
    if st.get_option("theme.base") == "dark":
        return "#B45309"
    return "#FDE68A"


def _reformat_inline_dates(text: str) -> str:
    """
    LibreOffice embeds dates in reply text as (MM/DD/YYYY, HH:MM).
    Reformat to match our display standard: Month D, YYYY · H:MM AM/PM
    Only applied when is_libreoffice=True is passed to render_comments.
    """

    def replace(m: re.Match) -> str:
        try:
            dt = datetime.strptime(m.group(0), "(%m/%d/%Y, %H:%M)")
            return f"({dt.strftime('%B %-d, %Y · %-I:%M %p')})"
        except ValueError:
            return m.group(0)

    return re.sub(r"\(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}\)", replace, text)


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
        st.markdown(f"> {html.escape(para)}", unsafe_allow_html=True)
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
        st.markdown(f"> {html.escape(para)}", unsafe_allow_html=True)
        return
    before = html.escape(para[:idx])
    after = html.escape(para[idx + len(text) :])
    escaped = html.escape(text)
    if kind == "insertion":
        styled = f'<span style="color:#3b82f6;text-decoration:underline">{escaped}</span>'
    else:
        styled = f'<span style="color:#ef4444;text-decoration:line-through">{escaped}</span>'
    st.markdown(f"{before}{styled}{after}", unsafe_allow_html=True)


def render_paragraph_with_highlight(para: str, selected: str) -> None:
    idx = para.find(selected)
    if idx == -1:
        st.markdown(f"> {para}")
        return
    before = para[:idx]
    after = para[idx + len(selected) :]
    annotated_text(before, (selected, "", _highlight_color()), after)


def render_comments(
    comments: list[Comment],
    version: WordVersion,
    order: list[str],
    is_libreoffice: bool = False,
) -> None:
    def render_elements(c: Comment) -> None:
        for label in order:
            match label:
                case "Sentence" if c.context:
                    for sent in c.context.sentences:
                        render_paragraph_with_highlight(
                            sent.text, c.context.selected_text
                        )
                case "Paragraph" if c.context:
                    with st.expander("📄 Paragraph"):
                        render_paragraph_with_highlight(
                            c.context.paragraph_text, c.context.selected_text
                        )
                case "Comment":
                    text = _reformat_inline_dates(c.text) if is_libreoffice else c.text
                    st.markdown(text)

    def render_reply(reply: Comment) -> None:
        status = "✅" if reply.resolved else "🔵"
        with st.container(border=True):
            st.caption(f"{status} **{reply.author}** · {_format_date(reply.date)}")
            render_elements(reply)

    st.caption(f"{len(comments)} comments · `{version.name}`")

    for i, comment in enumerate(comments, 1):
        status_badge = "✅ Resolved" if comment.resolved else "🔵 Open"
        with st.container(border=True):
            col_num, col_meta = st.columns([1, 11])
            with col_num:
                st.markdown(f"### {i}")
            with col_meta:
                st.markdown(
                    f"**{comment.author}** · {status_badge} · {_format_date(comment.date)}"
                )

            render_elements(comment)

            if comment.replies:
                n = len(comment.replies)
                with st.expander(f"↩️ {n} repl{'y' if n == 1 else 'ies'}"):
                    for reply in comment.replies:
                        render_reply(reply)
