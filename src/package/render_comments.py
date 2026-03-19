import streamlit as st
from annotated_text import annotated_text
from datetime import datetime, timezone
import zoneinfo
from .extract_comments import Comment, WordVersion


def _format_date(iso: str) -> str:
    try:
        # Strip the Z — Word writes local time but incorrectly marks it as UTC
        dt = datetime.fromisoformat(iso.rstrip("Z"))
        return dt.strftime("%-d %b %Y · %-I:%M %p")
    except ValueError:
        return iso


def _highlight_color() -> str:
    """Return a highlight color suited to the active Streamlit theme."""
    if st.get_option("theme.base") == "dark":
        return "#B45309"
    return "#FDE68A"


def render_comments(
    comments: list[Comment], version: WordVersion, order: list[str]
) -> None:
    def render_paragraph_with_highlight(para: str, selected: str) -> None:
        idx = para.find(selected)
        if idx == -1:
            st.markdown(f"> {para}")
            return
        before = para[:idx]
        after = para[idx + len(selected) :]
        annotated_text(before, (selected, "", _highlight_color()), after)

    def render_elements(c: Comment) -> None:
        for label in order:
            match label:
                case "Comment":
                    st.markdown(f"**Comment:**")
                    st.markdown(f"{c.text}")
                case "Paragraph" if c.context:
                    st.markdown("**Paragraph:**")
                    render_paragraph_with_highlight(
                        c.context.paragraph_text, c.context.selected_text
                    )
                case "Sentence" if c.context:
                    for sent in c.context.sentences:
                        st.markdown(f"**Sentence:**")
                        render_paragraph_with_highlight(sent, c.context.selected_text)

    st.markdown("# Comments")
    st.markdown(f"**Count:** {len(comments)}")
    st.divider()

    for i, comment in enumerate(comments, 1):
        status_badge = "✅ Resolved" if comment.resolved else "🔵 Open"
        st.markdown(
            f"### {i}. {comment.author}  ·  {status_badge}  ·  {_format_date(comment.date)}"
        )
        render_elements(comment)
        if comment.replies:
            n = len(comment.replies)
            with st.expander(f"↩️ {n} repl{'y' if n == 1 else 'ies'}"):
                for reply in comment.replies:
                    r_status = "✅" if reply.resolved else "↩️"
                    st.markdown(
                        f"{r_status} **{reply.author}** · {_format_date(reply.date)}"
                    )
                    render_elements(reply)
        st.divider()
