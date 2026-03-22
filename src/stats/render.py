from datetime import datetime

from src.comments.extract import Comment
from src.stats.compute import CommentMetrics
from src.stats.config import CFG
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

_DATE_FMT = CFG["display"]["date_format"]
_CARD_HEIGHT = CFG["chart"]["card_height"]
_MARKER_SIZE = CFG["chart"]["marker_size"]
_MARKER_OPACITY = CFG["chart"]["marker_opacity"]
_BAR_HEIGHT_PER_ROW = CFG["chart"]["bar_height_per_row"]
_BAR_HEIGHT_BASE = CFG["chart"]["bar_height_base"]
_TL_HEIGHT_PER_AUTHOR = CFG["chart"]["timeline_height_per_author"]
_TL_HEIGHT_BASE = CFG["chart"]["timeline_height_base"]


def _author_color_scale_from(authors: list[str]) -> alt.Scale:
    """Build a consistent color scale from an explicit author list."""
    colors = [_AUTHOR_PALETTE[i % len(_AUTHOR_PALETTE)] for i in range(len(authors))]
    return alt.Scale(domain=authors, range=colors)


def _author_color_map(authors: list[str]) -> dict[str, str]:
    sorted_authors = sorted(authors)
    return {
        a: _AUTHOR_PALETTE[i % len(_AUTHOR_PALETTE)]
        for i, a in enumerate(sorted_authors)
    }


# ---------------------------------------------------------------------------
# Date caption
# ---------------------------------------------------------------------------
def render_date_caption(
    df: pd.DataFrame, reference_date: datetime, is_closed: bool
) -> None:
    if df.empty:
        return
    earliest = df["date"].min().strftime(_DATE_FMT)
    end = (
        df["date"].max().strftime(_DATE_FMT)
        if is_closed
        else reference_date.strftime(_DATE_FMT)
    )
    st.caption(f"From {earliest} → {end}")


# ---------------------------------------------------------------------------
# Comment metrics
# ---------------------------------------------------------------------------
def render_comment_metrics(metrics: CommentMetrics, n_cols: int = 3) -> None:
    items = [
        ("Total", metrics.total, None),
        ("Open", metrics.open, None),
        ("Resolved", metrics.resolved, None),
    ]
    cols = st.columns(n_cols)
    for col, (label, value, delta) in zip(cols, items):
        tile = col.container(border=True, height=_CARD_HEIGHT)
        tile.metric(label, value, delta=delta)


# ---------------------------------------------------------------------------
# Thread depth
# ---------------------------------------------------------------------------
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
                scale=alt.Scale(domain=[True, False], range=["#22c55e", "#3b82f6"]),
            ),
            tooltip=[
                alt.Tooltip("author:N", title="Author"),
                alt.Tooltip("text:N", title="Comment"),
                alt.Tooltip("replies:Q", title="Replies"),
                alt.Tooltip("resolved:N", title="Resolved"),
            ],
        )
        .properties(
            title="Thread Depth",
            height=40 * len(df) + _BAR_HEIGHT_BASE,
        )
    )
    st.altair_chart(chart, width="stretch")


# ---------------------------------------------------------------------------
# Author bar
# ---------------------------------------------------------------------------
def render_author_bar(
    df: pd.DataFrame,
    title: str,
    all_authors: list[str] | None = None,
) -> None:
    if df.empty:
        st.caption(f"No data for {title}.")
        return

    authors = all_authors if all_authors else sorted(df["author"].unique().tolist())
    color_scale = _author_color_scale_from(authors)

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
        .properties(
            title=title,
            height=_BAR_HEIGHT_PER_ROW * len(counts) + _BAR_HEIGHT_BASE,
        )
    )
    st.altair_chart(chart, width="stretch")


# ---------------------------------------------------------------------------
# Comment timeline
# ---------------------------------------------------------------------------
def render_comment_timeline(
    df: pd.DataFrame,
    title: str,
    all_authors: list[str] | None = None,
    expanded_view_key: str = "_expanded_view",
    expand_all_key: str = "_expand_all",
    show_fields_key: str = "_show_fields",
    on_expanded_view=None,
    on_expand_all=None,
    on_show_fields=None,
) -> None:
    if df.empty:
        st.caption("No data matches the current filters.")
        return

    import numpy as np

    rng = np.random.default_rng(42)
    df = df.copy().reset_index(drop=True)

    visible_authors = sorted(df["author"].unique().tolist())
    color_authors = all_authors if all_authors else visible_authors
    color_map = _author_color_map(color_authors)
    author_idx = {a: i for i, a in enumerate(visible_authors)}

    df["jitter"] = rng.uniform(-0.3, 0.3, size=len(df))
    df["_idx"] = df.index
    df["y_jittered"] = df.apply(lambda r: author_idx[r["author"]] + r["jitter"], axis=1)

    fig = px.scatter(
        df,
        x="date",
        y="y_jittered",
        color="author",
        color_discrete_map=color_map,
        hover_data={
            "date": "|" + _DATE_FMT,
            "author": True,
            "kind": True,
            "resolved": True,
            "y_jittered": False,
            "_idx": True,
        },
        title=title,
        height=_TL_HEIGHT_PER_AUTHOR * len(visible_authors) + _TL_HEIGHT_BASE,
    )

    fig.update_layout(
        yaxis=dict(
            tickvals=list(range(len(visible_authors))),
            ticktext=visible_authors,
            title=None,
        ),
        xaxis_title="Date",
        dragmode="select",
        legend_title="Author",
        modebar_add=["lasso2d", "select2d"],
    )
    fig.update_traces(marker=dict(size=_MARKER_SIZE, opacity=_MARKER_OPACITY))

    event = st.plotly_chart(fig, on_select="rerun", width="stretch")
    has_selection = bool(event and event["selection"] and event["selection"]["points"])

    if has_selection:
        indices = [int(p["customdata"][-1]) for p in event["selection"]["points"]]
        selected = df.iloc[indices]
    else:
        selected = df

    if has_selection:
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
        [
            "author",
            "date",
            "kind",
            "resolved",
            "comment",
            "selected",
            "sentence",
            "paragraph",
        ]
    ].copy()
    display["sentence"] = pd.Series(display["sentence"]).apply(
        lambda s: " / ".join(s) if isinstance(s, list) else (s or "")
    )
    display["date"] = pd.to_datetime(pd.Series(display["date"])).dt.strftime(_DATE_FMT)
    display.columns = [c.capitalize() for c in display.columns]
    display = pd.DataFrame(display).sort_values("Date").reset_index(drop=True)

    col_view, col_expand = st.columns([1, 1])
    expanded_view = col_view.toggle(
        "Expanded view",
        key=expanded_view_key,
        on_change=on_expanded_view,
    )
    expand_all = (
        col_expand.toggle(
            "Expand all",
            key=expand_all_key,
            on_change=on_expand_all,
        )
        if expanded_view
        else False
    )

    ALL_FIELDS = ["Resolved", "Comment", "Selected", "Sentence", "Paragraph"]
    show_fields = st.multiselect(
        "Fields to show",
        options=ALL_FIELDS,
        key=show_fields_key,
        on_change=on_show_fields,
    )

    if expanded_view:
        for row in display.itertuples(index=False, name="Row"):  # type: ignore[assignment]
            if not show_fields:
                break
            with st.expander(
                f"{row.Author} · {row.Date} · {row.Kind}", expanded=expand_all
            ):

                def render_field(field: str) -> None:
                    match field:
                        case "Resolved":
                            st.markdown(
                                f"**Resolved:** {'Yes' if row.Resolved else 'No'}"
                            )
                        case "Comment":
                            st.markdown(f"**Comment:** {row.Comment}")
                        case "Selected" if row.Selected:
                            st.markdown("**Selected:**")
                            render_paragraph_with_highlight(row.Selected, row.Selected)
                        case "Sentence" if row.Sentence:
                            st.markdown("**Sentence:**")
                            sentences = (
                                row.Sentence
                                if isinstance(row.Sentence, list)
                                else [row.Sentence]
                            )
                            for sent in sentences:
                                render_paragraph_with_highlight(sent, row.Selected)
                        case "Paragraph" if row.Paragraph:
                            st.markdown("**Paragraph:**")
                            render_paragraph_with_highlight(row.Paragraph, row.Selected)

                for field in show_fields:
                    render_field(field)
    else:
        if show_fields:
            cols = ["Author", "Date", "Kind"] + [
                f for f in show_fields if f in display.columns
            ]
            st.dataframe(
                display[[c for c in cols if c in display.columns]],
                hide_index=True,
            )
