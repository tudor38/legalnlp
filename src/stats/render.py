from datetime import datetime
from typing import NamedTuple

from src.comments.extract import Comment
from src.stats.compute import CommentMetrics, PassageActivity
from src.stats.config import CFG
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from src.comments.render import render_paragraph_with_highlight

# Tableau10 desaturated — consistent across Altair and Plotly
AUTHOR_PALETTE = [
    "#7fa7c9",
    "#f5b97a",
    "#e88b8c",
    "#9dcdc9",
    "#8dbc8a",
    "#f2d97e",
    "#c9a3c4",
    "#ffbfc8",
    "#c0a08a",
    "#d0ccc8",
]

_DATE_FMT = CFG.display.date_format
_CARD_HEIGHT = CFG.chart.card_height
_MARKER_SIZE = CFG.chart.marker_size
_MARKER_OPACITY = CFG.chart.marker_opacity
_BAR_HEIGHT_PER_ROW = CFG.chart.bar_height_per_row
_BAR_HEIGHT_BASE = CFG.chart.bar_height_base
_TL_HEIGHT_PER_AUTHOR = CFG.chart.timeline_height_per_author
_TL_HEIGHT_BASE = CFG.chart.timeline_height_base


def _author_color_scale_from(authors: list[str]) -> alt.Scale:
    colors = [AUTHOR_PALETTE[i % len(AUTHOR_PALETTE)] for i in range(len(authors))]
    return alt.Scale(domain=authors, range=colors)


def _author_color_map(authors: list[str]) -> dict[str, str]:
    return {a: AUTHOR_PALETTE[i % len(AUTHOR_PALETTE)] for i, a in enumerate(authors)}


# ---------------------------------------------------------------------------
# Timeline field config
# ---------------------------------------------------------------------------
class TimelineField(NamedTuple):
    """Maps a display label to a DataFrame column and an optional highlight column."""

    label: str
    col: str
    highlight: str | None = None  # column to use for paragraph highlight


# Pre-built field configs for comments and redlines
COMMENT_FIELDS: list[TimelineField] = [
    TimelineField("Resolved", "Resolved", None),
    TimelineField("Comment", "Comment", None),
    TimelineField("Selected", "Selected", "Selected"),
    TimelineField("Sentence", "Sentence", "Selected"),
    TimelineField("Paragraph", "Paragraph", "Selected"),
]

REDLINE_FIELDS: list[TimelineField] = [
    TimelineField("Redline", "Text", None),
    TimelineField("Sentence", "Sentence", "Text"),
    TimelineField("Paragraph", "Paragraph", "Text"),
]


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
def render_comment_metrics(metrics: CommentMetrics, n_cols: int = 2) -> None:
    items = [
        ("Total", metrics.total, None),
        ("Open", metrics.open, None),
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
        st.caption(f"No data for `{title}`.")
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
# Problematic passages
# ---------------------------------------------------------------------------
def render_problematic_passages(passages: list[PassageActivity]) -> None:
    """
    Render a bar chart and ranked table of the most contested passages.
    Chart shows total activity per passage. Table shows full paragraph text
    with comment, redline, and move counts.
    """
    if not passages:
        st.caption("No activity found.")
        return

    df = pd.DataFrame(
        [
            {
                "rank": i + 1,
                "para_idx": p.para_idx,
                "paragraph": p.paragraph,
                "comments": p.comment_count,
                "redlines": p.redline_count,
                "moves": p.move_count,
                "total": p.total_activity,
                "authors": ", ".join(sorted(p.authors)),
            }
            for i, p in enumerate(passages)
        ]
    )

    # --- Bar chart ---
    chart_df = df.copy()
    chart_df["label"] = (
        "#" + chart_df["rank"].astype(str) + "  " + chart_df["paragraph"].str[:60] + "…"
    )

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "total:Q",
                title="Total Activity",
                axis=alt.Axis(tickMinStep=1, format="d"),
            ),
            y=alt.Y("label:N", sort="-x", title=None),
            color=alt.value(AUTHOR_PALETTE[0]),
            tooltip=[
                alt.Tooltip("rank:Q", title="Rank"),
                alt.Tooltip("total:Q", title="Total Activity"),
                alt.Tooltip("comments:Q", title="Comments"),
                alt.Tooltip("redlines:Q", title="Redlines"),
                alt.Tooltip("moves:Q", title="Moves"),
                alt.Tooltip("authors:N", title="Authors"),
            ],
        )
        .properties(
            title="Most Contested Passages",
            height=_BAR_HEIGHT_PER_ROW * len(df) + _BAR_HEIGHT_BASE,
        )
    )
    st.altair_chart(chart, width="stretch")

    # --- Ranked table ---
    st.markdown("#### Passage Detail")
    for _, row in df.iterrows():
        with st.expander(
            f"#{int(row['rank'])}  ·  {int(row['total'])} activity  ·  "
            f"{int(row['comments'])} comments  ·  {int(row['redlines'])} redlines  ·  "
            f"{int(row['moves'])} moves",
            expanded=False,
        ):
            st.markdown(f"**Authors:** {row['authors']}")
            st.markdown("**Paragraph:**")
            st.markdown(f"> {row['paragraph']}")


# ---------------------------------------------------------------------------
# Generalized timeline
# ---------------------------------------------------------------------------
def render_timeline(
    df: pd.DataFrame,
    title: str,
    fields: list[TimelineField],
    display_cols: list[str],  # df columns to include in table
    default_fields: list[str],  # default multiselect selection
    all_authors: list[str] | None = None,
    expanded_view_key: str = "_expanded_view",
    expand_all_key: str = "_expand_all",
    show_fields_key: str = "_show_fields",
    on_expanded_view=None,
    on_expand_all=None,
    on_show_fields=None,
) -> None:
    """
    Generalized timeline scatter + detail table.

    Parameters
    ----------
    fields         : list of TimelineField defining available fields for the
                     expanded view and multiselect options
    display_cols   : df columns to pull into the display table
    default_fields : which field labels are selected by default
    """
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

    hover_data: dict = {
        "date": "|" + _DATE_FMT,
        "author": True,
        "y_jittered": False,
    }
    if "resolved" in df.columns:
        hover_data["resolved"] = True
    if "kind" in df.columns:
        hover_data["kind"] = True
    hover_data["_idx"] = True

    fig = px.scatter(
        df,
        x="date",
        y="y_jittered",
        color="author",
        color_discrete_map=color_map,
        hover_data=hover_data,
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
        authors_sel = int(selected["author"].nunique())

        cols = st.columns(2)
        cols[0].metric("Selected", total_sel)
        cols[1].metric("Authors", authors_sel)

    # Build display table
    valid_cols = [c for c in display_cols if c in df.columns]
    display = selected[valid_cols].copy()

    # Join sentence lists to string
    if "sentence" in display.columns:
        display["sentence"] = pd.Series(display["sentence"]).apply(
            lambda s: " / ".join(s) if isinstance(s, list) else (s or "")
        )

    display["date"] = pd.to_datetime(pd.Series(display["date"])).dt.strftime(_DATE_FMT)
    display.columns = [c.capitalize() for c in display.columns]
    display = pd.DataFrame(display).sort_values("Date").reset_index(drop=True)

    # Table controls
    col_view, col_expand = st.columns([1, 1])
    expanded_view = col_view.toggle(
        "Show expanded view",
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

    all_field_labels = [f.label for f in fields]
    show_fields = st.multiselect(
        "Fields to show",
        options=all_field_labels,
        key=show_fields_key,
        on_change=on_show_fields,
    )

    # Build label → TimelineField lookup
    field_map = {f.label: f for f in fields}

    if expanded_view:
        for _, row in display.iterrows():
            if not show_fields:
                break
            kind_label = f" · {row['Kind']}" if "Kind" in display.columns else ""
            with st.expander(
                f"{row['Author']} · {row['Date']}{kind_label}", expanded=expand_all
            ):
                for label in show_fields:
                    tf = field_map.get(label)
                    if tf is None:
                        continue
                    col_name = tf.col
                    hl_name = tf.highlight

                    if col_name not in display.columns:
                        continue

                    val = row[col_name]
                    if not val and val != 0:
                        continue

                    if col_name == "Resolved":
                        st.markdown(f"**Resolved:** {'Yes' if val else 'No'}")
                    elif hl_name and hl_name in display.columns:
                        hl_val = row[hl_name]
                        st.markdown(f"**{label}:**")
                        items = val if isinstance(val, list) else [val]
                        for item in items:
                            render_paragraph_with_highlight(item, hl_val or "")
                    else:
                        st.markdown(f"**{label}:** {val}")
    else:
        if show_fields:
            keep_cols = ["Author", "Date"]
            if "Kind" in display.columns:
                keep_cols.append("Kind")
            for label in show_fields:
                tf = field_map.get(label)
                if tf and tf.col.capitalize() in display.columns:
                    keep_cols.append(tf.col.capitalize())
            st.dataframe(
                display[[c for c in keep_cols if c in display.columns]],
                hide_index=True,
            )
