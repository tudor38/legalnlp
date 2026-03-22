from src.comments.extract import Comment
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from src.comments.render import render_paragraph_with_highlight

_AUTHOR_PALETTE = [
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


def _author_color_scale(df: pd.DataFrame) -> alt.Scale:
    authors = sorted(df["author"].unique().tolist())
    colors = [_AUTHOR_PALETTE[i % len(_AUTHOR_PALETTE)] for i in range(len(authors))]
    return alt.Scale(domain=authors, range=colors)


def _author_color_map(authors: list[str]) -> dict[str, str]:
    """Return {author: color} using the same palette as _author_color_scale."""
    sorted_authors = sorted(authors)
    return {
        a: _AUTHOR_PALETTE[i % len(_AUTHOR_PALETTE)]
        for i, a in enumerate(sorted_authors)
    }


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


def render_author_bar(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        st.caption(f"No data for {title}.")
        return

    color_scale = _author_color_scale(df)

    counts = (
        df.groupby("author")
        .size()
        .to_frame("count")
        .reset_index()
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


def render_comment_timeline(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        st.caption(f"No data for {title}.")
        return

    import numpy as np

    rng = np.random.default_rng(42)
    df = df.copy().reset_index(drop=True)
    df["jitter"] = rng.uniform(-0.3, 0.3, size=len(df))
    df["_idx"] = df.index

    all_authors = sorted(df["author"].unique().tolist())
    color_map = _author_color_map(all_authors)
    author_idx = {a: i for i, a in enumerate(all_authors)}
    df["y_jittered"] = df.apply(lambda r: author_idx[r["author"]] + r["jitter"], axis=1)

    fig = px.scatter(
        df,
        x="date",
        y="y_jittered",
        color="author",
        color_discrete_map=color_map,
        hover_data={
            "date": "|%B %d, %Y",
            "author": True,
            "kind": True,
            "resolved": True,
            "y_jittered": False,
            "_idx": True,
        },
        title=title,
        height=60 * len(all_authors) + 120,
    )

    fig.update_layout(
        yaxis=dict(
            tickvals=list(range(len(all_authors))),
            ticktext=all_authors,
            title=None,
        ),
        xaxis_title="Date",
        dragmode="select",
        legend_title="Author",
        modebar_add=["lasso2d", "select2d"],
    )
    fig.update_traces(marker=dict(size=10, opacity=0.75))

    event = st.plotly_chart(fig, on_select="rerun", width="stretch")

    if event and event["selection"] and event["selection"]["points"]:
        indices = [int(p["customdata"][-1]) for p in event["selection"]["points"]]
        selected = df.iloc[indices]
    else:
        selected = df

    total_sel = len(selected)
    resolved_sel = int(selected["resolved"].sum())
    authors_sel = int(selected["author"].nunique())
    comments_sel = int((selected["kind"] == "comment").sum())
    replies_sel = int((selected["kind"] == "reply").sum())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Selected", total_sel)
    col2.metric("Authors", authors_sel)
    col3.metric("Comments", comments_sel)
    col4.metric("Replies", replies_sel)
    col5.metric("Resolved", f"{resolved_sel}")

    display = selected[
        ["author", "date", "kind", "resolved", "comment", "selected", "sentence", "paragraph"]
    ].copy()

    display["date"] = pd.to_datetime(pd.Series(display["date"])).dt.strftime(
        "%B %-d, %Y"
    )
    display.columns = [c.capitalize() for c in display.columns]
    display = pd.DataFrame(display).sort_values("Date").reset_index(drop=True)
    col_view, col_expand = st.columns([1, 1])
    expanded_view = col_view.toggle("Expanded view", value=False)
    expand_all    = col_expand.toggle("Expand all", value=False) if expanded_view else False

    ALL_FIELDS     = ["Resolved", "Comment", "Selected", "Sentence", "Paragraph"]
    DEFAULT_FIELDS = ["Resolved", "Comment", "Sentence"]
    show_fields    = st.multiselect(
        "Fields to show",
        options=ALL_FIELDS,
        default=DEFAULT_FIELDS,
    )

    if expanded_view:
        for row in display.itertuples(index=False, name="Row"):  # type: ignore[assignment]
            if not show_fields:
                break
            with st.expander(f"{row.Author} · {row.Date} · {row.Kind}", expanded=expand_all):
                if "Resolved"  in show_fields: st.markdown(f"**Resolved:** {'Yes' if row.Resolved else 'No'}")
                if "Comment"      in show_fields: st.markdown(f"**Comment:** {row.Comment}")
                if "Selected"  in show_fields and row.Selected:
                    st.markdown("**Selected:**")
                    render_paragraph_with_highlight(row.Selected, row.Selected)
                if "Sentence" in show_fields and row.Sentence:
                    st.markdown("**Sentence:**")
                    sentences = row.Sentence if isinstance(row.Sentence, list) else [row.Sentence]
                    for sent in sentences:
                        render_paragraph_with_highlight(sent, row.Selected)
                if "Paragraph" in show_fields and row.Paragraph:
                    st.markdown("**Paragraph:**")
                    render_paragraph_with_highlight(row.Paragraph, row.Selected)
    else:
        if show_fields:
            cols = ["Author", "Date", "Kind"] + [f for f in show_fields if f in display.columns]
            st.dataframe(display[[c for c in cols if c in display.columns]], hide_index=True)
