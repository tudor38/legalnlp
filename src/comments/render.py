import re
import streamlit as st
import pandas as pd
from annotated_text import annotated_text
from datetime import datetime
from src.comments.extract import Comment, WordVersion
import altair as alt


def render_thread_depth(comments: list[Comment]) -> None:
    df = pd.DataFrame([c.to_row() for c in comments if c.replies])
    if df.empty:
        st.caption("No threaded comments.")
        return

    df["resolved"] = df["resolved"].replace({True: "Yes", False: "No"})

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "replies:Q", title="Replies", axis=alt.Axis(tickMinStep=1, format="d")
            ),
            y=alt.Y("author:N", sort="-x", title="Author"),
            color=alt.Color(
                "resolved:N",
                scale=alt.Scale(domain=["No", "Yes"], range=["#ff4b4b", "#21c354"]),
                legend=alt.Legend(title="Resolved"),
            ),
            tooltip=[
                alt.Tooltip("author:N", title="Author"),
                alt.Tooltip("text:N", title="Comment"),
                alt.Tooltip("replies:Q", title="Replies"),
                alt.Tooltip("resolved:N", title="Resolved"),
            ],
        )
        .properties(
            title="Comments with most replies, possibly indicating disagreement or complex issues",
            # height=60 * len(df) + 60,
        )
    )
    st.altair_chart(chart, width="stretch")


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
