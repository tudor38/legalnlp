from src.comments.extract import Comment
from src.stats.compute import open_comment_ages
import streamlit as st
import pandas as pd
import altair as alt


def render_thread_depth(comments: list[Comment]) -> None:
    df = pd.DataFrame([c.to_row() for c in comments if c.replies])

    if df.empty:
        st.caption("No threaded comments.")
        return

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("replies:Q", title="Replies"),
            y=alt.Y("author:N", sort="-x", title="Author"),
            color=alt.Color("resolved:N", scale=alt.Scale(
                domain=[True, False],
                range=["#22c55e", "#3b82f6"],
            )),
            tooltip=[
                alt.Tooltip("author:N",  title="Author"),
                alt.Tooltip("text:N",    title="Comment"),
                alt.Tooltip("replies:Q", title="Replies"),
                alt.Tooltip("resolved:N", title="Resolved"),
            ],
        )
        .properties(title="Thread Depth", height=40 * len(df) + 60)
    )

    st.altair_chart(chart, use_container_width=True)

def render_resolution_rate(comments: list[Comment]) -> float:
    df       = pd.DataFrame([c.to_row() for c in comments])
    total    = len(df)
    resolved = df["resolved"].sum()
    rate     = resolved / total if total else 0.0

    st.metric(
        label="Resolution Rate",
        value=f"{rate:.0%}",
        delta=f"{resolved} of {total} resolved",
    )

    return rate

def render_open_comment_ages(comments: list[Comment]) -> None:
    df = open_comment_ages(comments)
    if df.empty:
        st.caption("No open comments.")
        return

    primary = st.get_option("theme.primaryColor") or "#ff4b4b"

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("age_days:Q", title="Age (days)", axis=alt.Axis(tickMinStep=1, format="d")),
            y=alt.Y("author:N", sort="-x", title="Author"),
            color=alt.Color("age_days:Q",
                scale=alt.Scale(range=["#444444", primary]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("author:N",   title="Author"),
                alt.Tooltip("text:N",     title="Comment"),
                alt.Tooltip("age_days:Q", title="Days Open"),
            ],
        )
        .properties(title="Age of open comments", height=60 * len(df) + 60)
    )
    st.altair_chart(chart, use_container_width=True)
