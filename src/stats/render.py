from src.comments.extract import Comment
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

from src.stats.compute import CommentSummary
from src.stats.compute import RedlineSummary


def _author_color_scale(df: pd.DataFrame) -> alt.Scale:
    """Build a consistent color scale across all charts for the same dataset."""
    authors = sorted(df["author"].unique().tolist())
    # Altair's default categorical palette
    palette = [
        "#4c78a8",
        "#f58518",
        "#e45756",
        "#72b7b2",
        "#54a24b",
        "#eeca3b",
        "#b279a2",
        "#ff9da6",
        "#9d755d",
        "#bab0ac",
    ]
    colors = [palette[i % len(palette)] for i in range(len(authors))]
    return alt.Scale(domain=authors, range=colors)


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
            color=alt.Color(
                "resolved:N",
                scale=alt.Scale(
                    domain=[True, False],
                    range=["#22c55e", "#3b82f6"],
                ),
            ),
            tooltip=[
                alt.Tooltip("author:N", title="Author"),
                alt.Tooltip("text:N", title="Comment"),
                alt.Tooltip("replies:Q", title="Replies"),
                alt.Tooltip("resolved:N", title="Resolved"),
            ],
        )
        .properties(title="Thread Depth", height=40 * len(df) + 60)
    )

    st.altair_chart(chart, width="stretch")


def render_resolution_rate(comments: list[Comment]) -> float:
    df = pd.DataFrame([c.to_row() for c in comments])
    total = len(df)
    resolved = df["resolved"].sum()
    rate = resolved / total if total else 0.0

    st.metric(
        label="Resolution Rate",
        value=f"{rate:.0%}",
        delta=f"{resolved} of {total} resolved",
    )

    return rate


def render_age_boxplot(
    df: pd.DataFrame,
    title: str,
) -> None:

    if df.empty:
        st.caption(f"No data for {title}.")
        return

    color_scale = _author_color_scale(df)

    chart = (
        alt.Chart(df)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("age_days:Q", title="Age (days)"),
            y=alt.Y("author:N", title=None),
            color=alt.Color("author:N", scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip("author:N", title="Author"),
                alt.Tooltip("age_days:Q", title="Age (days)"),
            ],
        )
        .properties(height=50 * df["author"].nunique() + 60)
    )
    st.markdown(
        "**Age Distribution**",
        help=(
            "This plot maps the **negotiation lifecycle**, highlighting the start-to-finish duration and "
            "the 'middle 50%' where the most intensive revision activity occurred.\n\n"
            "**What each element shows:**\n"
            "- **Box:** the middle 50% of item ages for that reviewer\n"
            "- **Line through the box:** the median age\n"
            "- **Whiskers:** the oldest and newest items\n"
            "- **Dots beyond the whiskers:** outliers, items unusually old or new\n\n"
            "**What to look for:**\n"
            "- **Wide boxes** signal activity spread over a long period\n"
            "- **Outliers** may represent forgotten or contested items\n"
            "- **Non-overlapping boxes** across reviewers indicate sequential rather than parallel review"
        ),
    )
    st.altair_chart(chart, width="stretch")


def render_author_bar(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        st.caption(f"No data for {title}.")
        return

    color_scale = _author_color_scale(df)

    counts = (
        df.groupby("author")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Count", axis=alt.Axis(tickMinStep=1, format="d")),
            y=alt.Y("author:N", sort="-x", title=None),
            color=alt.Color("author:N", scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip("author:N", title="Author"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(title=title, height=50 * len(counts) + 60)
    )
    st.altair_chart(chart, width="stretch")


def render_comment_metrics(
    total: int,
    resolved: int,
    n_cols: int = 3,
) -> None:
    open_ = total - resolved

    items = [
        ("Total", total, None),
        ("Open", open_, None),
        ("Resolved", resolved, None),
    ]

    cols = st.columns(n_cols)
    for col, (label, value, delta) in zip(cols, items):
        tile = col.container(border=True, height=120)
        tile.metric(label, value, delta=delta)
