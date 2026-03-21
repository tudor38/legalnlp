from src.comments.extract import Comment, DocumentParagraphs
from src.stats.compute import open_comment_ages, paragraph_comment_density
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

from src.stats.compute import CommentSummary
import json
import streamlit.components.v1 as components
from src.stats.compute import RedlineSummary


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

    st.altair_chart(chart, width='stretch')


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


def render_comment_summary(summary: CommentSummary) -> None:
    from datetime import datetime

    def fmt_date(dt: datetime | None) -> str:
        return dt.strftime("%B %-d, %Y") if dt else "—"

    def fmt_days(d: float | None) -> str:
        return f"{d:.0f}" if d is not None else "—"

    # --- Resolution rate donut ---
    donut_df = pd.DataFrame(
        [
            {"status": "Resolved", "value": summary.resolved},
            {"status": "Unresolved", "value": summary.open},
        ]
    )

    donut = (
        alt.Chart(donut_df)
        .mark_arc(innerRadius=50, outerRadius=80)
        .encode(
            theta=alt.Theta("value:Q"),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(
                    domain=["Resolved", "Unresolved"],
                    range=["#21c354", "#374151"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("status:N", title="Status"),
                alt.Tooltip("value:Q", title="Count"),
            ],
        )
        .properties(
            width=220, height=220, title=f"{summary.resolution_rate:.0%} Resolved"
        )
    )

    # --- Stat pills ---
    pills_html = f"""
    <style>
    .pill-grid {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 8px;
    }}
    .pill {{
        background: var(--background-color, #f0f2f6);
        border-radius: 12px;
        padding: 14px 20px;
        min-width: 130px;
        flex: 1;
    }}
    .pill-label {{
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }}
    .pill-value {{
        font-size: 26px;
        font-weight: 600;
        color: var(--text-color, #111);
        line-height: 1.1;
    }}
    .pill-unit {{
        font-size: 13px;
        color: #aaa;
        margin-left: 4px;
    }}
    </style>
    <div class="pill-grid">
        <div class="pill">
            <div class="pill-label">Total</div>
            <div class="pill-value">{summary.total}</div>
        </div>
        <div class="pill">
            <div class="pill-label">Resolved</div>
            <div class="pill-value" style="color:#21c354">{summary.resolved}</div>
        </div>
        <div class="pill">
            <div class="pill-label">Open</div>
            <div class="pill-value" style="color:#ff4b4b">{summary.open}</div>
        </div>
        <div class="pill">
            <div class="pill-label">Authors</div>
            <div class="pill-value">{summary.total_authors}</div>
        </div>
        <div class="pill">
            <div class="pill-label">Span</div>
            <div class="pill-value">{fmt_days(summary.span_days)}<span class="pill-unit">days</span></div>
        </div>
        <div class="pill">
            <div class="pill-label">Avg Age</div>
            <div class="pill-value">{fmt_days(summary.avg_age_days)}<span class="pill-unit">days</span></div>
        </div>
        <div class="pill">
            <div class="pill-label">Avg Open Age</div>
            <div class="pill-value">{fmt_days(summary.avg_open_age_days)}<span class="pill-unit">days</span></div>
        </div>
    </div>
    """

    # --- Date range timeline bar ---
    timeline_html = f"""
    <div style="margin-top:20px; padding: 14px 20px; background: var(--background-color, #f0f2f6); border-radius:12px;">
        <div style="font-size:12px; color:#888; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;">Comment Period</div>
        <div style="display:flex; align-items:center; gap:16px;">
            <div>
                <div style="font-size:11px; color:#aaa;">Earliest</div>
                <div style="font-size:16px; font-weight:600;">{fmt_date(summary.earliest_comment)}</div>
            </div>
            <div style="flex:1; height:3px; background: linear-gradient(to right, #21c354, #ff4b4b); border-radius:2px;"></div>
            <div style="text-align:right;">
                <div style="font-size:11px; color:#aaa;">Latest</div>
                <div style="font-size:16px; font-weight:600;">{fmt_date(summary.latest_comment)}</div>
            </div>
        </div>
    </div>
    """

    col_donut, col_stats = st.columns([2, 5])
    with col_donut:
        st.altair_chart(donut, width='content')
    with col_stats:
        st.markdown(pills_html, unsafe_allow_html=True)
        st.markdown(timeline_html, unsafe_allow_html=True)


def render_redline_summary(summary: RedlineSummary) -> None:
    from datetime import datetime

    def fmt_date(dt: datetime | None) -> str:
        return dt.strftime("%B %-d, %Y") if dt else "—"

    def fmt_days(d: float | None) -> str:
        return f"{d:.0f}" if d is not None else "—"

    donut_df = pd.DataFrame([
        {"kind": "Redlined",   "value": summary.redlined_chars},
        {"kind": "Unredlined", "value": max(0, summary.total_chars - summary.redlined_chars)},
    ])

    donut = (
        alt.Chart(donut_df)
        .mark_arc(innerRadius=50, outerRadius=80)
        .encode(
            theta=alt.Theta("value:Q"),
            color=alt.Color("kind:N",
                scale=alt.Scale(
                    domain=["Redlined", "Unredlined"],
                    range=["#ff4b4b", "#374151"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("kind:N",  title="Kind"),
                alt.Tooltip("value:Q", title="Characters"),
            ],
        )
        .properties(width=220, height=220, title=f"{summary.redline_density:.0%} Redlined")
    )

    pills_html = f"""
    <style>
    .pill-grid {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 8px;
    }}
    .pill {{
        background: var(--background-color, #f0f2f6);
        border-radius: 12px;
        padding: 14px 20px;
        min-width: 130px;
        flex: 1;
    }}
    .pill-label {{
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }}
    .pill-value {{
        font-size: 26px;
        font-weight: 600;
        color: var(--text-color, #111);
        line-height: 1.1;
    }}
    .pill-unit {{
        font-size: 13px;
        color: #aaa;
        margin-left: 4px;
    }}
    </style>
    <div class="pill-grid">
        <div class="pill">
            <div class="pill-label">Total</div>
            <div class="pill-value">{summary.total}</div>
        </div>
        <div class="pill">
            <div class="pill-label">Insertions</div>
            <div class="pill-value" style="color:#21c354">{summary.insertions}</div>
        </div>
        <div class="pill">
            <div class="pill-label">Deletions</div>
            <div class="pill-value" style="color:#ff4b4b">{summary.deletions}</div>
        </div>
        <div class="pill">
            <div class="pill-label">Authors</div>
            <div class="pill-value">{summary.total_authors}</div>
        </div>
        <div class="pill">
            <div class="pill-label">Span</div>
            <div class="pill-value">{fmt_days(summary.span_days)}<span class="pill-unit">days</span></div>
        </div>
        <div class="pill">
            <div class="pill-label">Avg Age</div>
            <div class="pill-value">{fmt_days(summary.avg_age_days)}<span class="pill-unit">days</span></div>
        </div>
        <div class="pill">
            <div class="pill-label">Redlined</div>
            <div class="pill-value">{summary.redline_density:.0%}</div>
        </div>
    </div>
    """

    timeline_html = f"""
    <div style="margin-top:20px; padding: 14px 20px; background: var(--background-color, #f0f2f6); border-radius:12px;">
        <div style="font-size:12px; color:#888; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;">Redline Period</div>
        <div style="display:flex; align-items:center; gap:16px;">
            <div>
                <div style="font-size:11px; color:#aaa;">Earliest</div>
                <div style="font-size:16px; font-weight:600;">{fmt_date(summary.earliest_redline)}</div>
            </div>
            <div style="flex:1; height:3px; background: linear-gradient(to right, #21c354, #ff4b4b); border-radius:2px;"></div>
            <div style="text-align:right;">
                <div style="font-size:11px; color:#aaa;">Latest</div>
                <div style="font-size:16px; font-weight:600;">{fmt_date(summary.latest_redline)}</div>
            </div>
        </div>
    </div>
    """

    col_donut, col_stats = st.columns([2, 5])
    with col_donut:
        st.altair_chart(donut, width='content')
    with col_stats:
        st.markdown(pills_html, unsafe_allow_html=True)
        st.markdown(timeline_html, unsafe_allow_html=True)


# def render_comment_timeline — commented out in original, preserved below
# (omitted for brevity, unchanged from original)
