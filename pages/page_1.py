import io
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd

from src.comments.extract import extract_comments, extract_paragraphs
from src.redlines.extract import extract_redlines, extract_moves
from src.stats.compute import (
    comment_ages_df,
    redline_ages_df,
    move_ages_df,
)
from src.stats.render import (
    render_author_bar,
    render_comment_metrics,
    render_comment_timeline,
)


# ---------------------------------------------------------------------------
# Session state — initialise permanent keys once
# ---------------------------------------------------------------------------
DEFAULTS: dict = {
    "p1_is_closed": False,
    "p1_closed_date": None,
    "p1_file_bytes": None,
    "p1_file_name": None,
    "p1_expanded_view": False,
    "p1_expand_all": False,
    "p1_show_fields": ["Resolved", "Sentence", "Comment"],
    "p1_timeline_authors": [],
    "p1_date_min": None,
    "p1_date_max": None,
    "p1_main_tab": "Comments",
    "p1_comment_tab": "Counts",
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def _store_uploaded_file():
    f = st.session_state["_p1_uploaded_file"]
    if f is not None:
        st.session_state["p1_file_bytes"] = f.read()
        st.session_state["p1_file_name"] = f.name
        f.seek(0)
    else:
        st.session_state["p1_file_bytes"] = None
        st.session_state["p1_file_name"] = None


def _store_is_closed():
    st.session_state["p1_is_closed"] = st.session_state["_p1_is_closed"]


def _store_closed_date():
    st.session_state["p1_closed_date"] = st.session_state["_p1_closed_date"]


def _store_expanded_view():
    st.session_state["p1_expanded_view"] = st.session_state["_p1_expanded_view"]


def _store_expand_all():
    st.session_state["p1_expand_all"] = st.session_state["_p1_expand_all"]


def _store_show_fields():
    st.session_state["p1_show_fields"] = st.session_state["_p1_show_fields"]


def _store_timeline_authors():
    st.session_state["p1_timeline_authors"] = st.session_state["_p1_timeline_authors"]


def _store_main_tab():
    st.session_state["p1_main_tab"] = st.session_state["_p1_main_tab"]


def _store_comment_tab():
    st.session_state["p1_comment_tab"] = st.session_state["_p1_comment_tab"]


def _store_date_range():
    val = st.session_state["_p1_date_range"]
    st.session_state["p1_date_min"] = val[0]
    st.session_state["p1_date_max"] = val[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _latest_date(comments, redlines) -> datetime | None:
    dates = []
    for c in comments:
        try:
            dates.append(datetime.fromisoformat(c.date.rstrip("Z")))
        except ValueError:
            pass
    for r in redlines:
        try:
            dates.append(datetime.fromisoformat(r.date.rstrip("Z")))
        except ValueError:
            pass
    return max(dates) if dates else None


def _load_document(file_bytes: bytes) -> tuple:
    comments, version = extract_comments(io.BytesIO(file_bytes))
    redlines, _ = extract_redlines(io.BytesIO(file_bytes))
    moves, _ = extract_moves(io.BytesIO(file_bytes))
    doc_paragraphs = extract_paragraphs(io.BytesIO(file_bytes))
    return comments, version, redlines, moves, doc_paragraphs


def _filter_by_date(df: pd.DataFrame, date_min, date_max) -> pd.DataFrame:
    """Filter a DataFrame with a 'date' column by date range."""
    if df.empty:
        return df
    return df[
        (df["date"].dt.date >= date_min) & (df["date"].dt.date <= date_max)
    ].reset_index(drop=True)


def _sidebar_controls(
    comments, redlines, c_df, r_df, m_df
) -> tuple[datetime, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Render all sidebar controls.

    Returns
    -------
    reference_date         : datetime
    filtered_c_df          : comments filtered by author + date range
    filtered_r_df          : redlines filtered by date range
    filtered_m_df          : moves filtered by date range
    """
    # --- Matter closed ---
    st.sidebar.markdown("### Document")
    st.session_state["_p1_is_closed"] = st.session_state["p1_is_closed"]
    is_closed = st.sidebar.toggle(
        "Matter is closed",
        key="_p1_is_closed",
        value=False,
        on_change=_store_is_closed,
        help="Toggle ON if the document is part of a closed matter. This setting affects date calculations.",
    )

    if is_closed:
        latest = _latest_date(comments, redlines)
        default = latest.date() if latest else datetime.today().date()
        saved = st.session_state["p1_closed_date"]
        st.session_state["_p1_closed_date"] = saved if saved is not None else default
        closed_date = st.sidebar.date_input(
            "Closed date",
            key="_p1_closed_date",
            on_change=_store_closed_date,
        )
        reference_date = datetime.combine(closed_date, datetime.min.time()) + timedelta(
            days=1
        )
    else:
        reference_date = datetime.now()

    all_dfs = [df for df in [c_df, r_df, m_df] if not df.empty]
    if not all_dfs:
        return reference_date, c_df, r_df, m_df

    # --- Global date range ---
    st.sidebar.markdown("### Filters")

    global_date_min = min(df["date"].min().date() for df in all_dfs)
    global_date_max = max(df["date"].max().date() for df in all_dfs)

    if not is_closed:
        global_date_max = datetime.now().date()

    saved_min = st.session_state["p1_date_min"] or global_date_min
    saved_max = st.session_state["p1_date_max"] or global_date_max
    saved_min = max(saved_min, global_date_min)
    saved_max = min(saved_max, global_date_max)

    date_range = st.sidebar.slider(
        "Date range",
        min_value=global_date_min,
        max_value=global_date_max,
        value=(saved_min, saved_max),
        key="_p1_date_range",
        on_change=_store_date_range,
    )

    # --- Author filter (comments only) ---
    if not c_df.empty:
        all_authors = sorted(c_df["author"].unique().tolist())
        if not st.session_state["p1_timeline_authors"]:
            st.session_state["_p1_timeline_authors"] = all_authors
        else:
            st.session_state["_p1_timeline_authors"] = st.session_state[
                "p1_timeline_authors"
            ]

        selected_authors = (
            st.sidebar.multiselect(
                "Comment authors",
                options=all_authors,
                key="_p1_timeline_authors",
                on_change=_store_timeline_authors,
            )
            or all_authors
        )
    else:
        selected_authors = []

    # --- Apply filters ---
    filtered_c_df = _filter_by_date(c_df, date_range[0], date_range[1])
    if not filtered_c_df.empty and selected_authors:
        filtered_c_df = filtered_c_df[
            filtered_c_df["author"].isin(selected_authors)
        ].reset_index(drop=True)

    filtered_r_df = _filter_by_date(r_df, date_range[0], date_range[1])
    filtered_m_df = _filter_by_date(m_df, date_range[0], date_range[1])

    return reference_date, filtered_c_df, filtered_r_df, filtered_m_df


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.sidebar.file_uploader(
    "Choose a file",
    type=["docx", "doc"],
    key="_p1_uploaded_file",
    on_change=_store_uploaded_file,
)

file_bytes = st.session_state["p1_file_bytes"]

if file_bytes:
    comments, version, redlines, moves, doc_paragraphs = _load_document(file_bytes)

    # Initial load with datetime.now() to get date ranges for sidebar
    c_df = comment_ages_df(comments, datetime.now())
    r_df = redline_ages_df(redlines, datetime.now())
    m_df = move_ages_df(moves, datetime.now())

    reference_date, filtered_c_df, filtered_r_df, filtered_m_df = _sidebar_controls(
        comments, redlines, c_df, r_df, m_df
    )

    # Recompute with correct reference_date then re-apply filters
    c_df = comment_ages_df(comments, reference_date)
    r_df = redline_ages_df(redlines, reference_date)
    m_df = move_ages_df(moves, reference_date)

    date_min = st.session_state.get("p1_date_min")
    date_max = st.session_state.get("p1_date_max")

    if date_min and date_max and not c_df.empty:
        global_date_min = min(
            df["date"].min().date() for df in [c_df, r_df, m_df] if not df.empty
        )
        global_date_max = max(
            df["date"].max().date() for df in [c_df, r_df, m_df] if not df.empty
        )
        date_min = max(date_min, global_date_min)
        date_max = min(date_max, global_date_max)

        sel_authors = st.session_state.get("p1_timeline_authors") or sorted(
            c_df["author"].unique().tolist()
        )
        filtered_c_df = _filter_by_date(c_df, date_min, date_max)
        filtered_c_df = filtered_c_df[
            filtered_c_df["author"].isin(sel_authors)
        ].reset_index(drop=True)
        filtered_r_df = _filter_by_date(r_df, date_min, date_max)
        filtered_m_df = _filter_by_date(m_df, date_min, date_max)

    # --- Main tab selection (persistent) ---
    st.session_state["_p1_main_tab"] = st.session_state["p1_main_tab"]
    main_tab = st.pills(
        "Section",
        ["Comments", "Redlines", "Moves"],
        key="_p1_main_tab",
        on_change=_store_main_tab,
        selection_mode="single",
    )

    # ---------------------------------------------------------------------------
    # Comments
    # ---------------------------------------------------------------------------
    if main_tab == "Comments":
        if not c_df.empty:
            earliest = (
                filtered_c_df["date"].min().strftime("%B %-d, %Y")
                if not filtered_c_df.empty
                else "—"
            )
            end = reference_date.strftime("%B %-d, %Y")
            st.caption(f"From {earliest} → {end}")

        total = len(comments)
        replies = sum(len(c.replies) for c in comments)
        resolved = sum(1 for c in comments if c.resolved) + sum(
            1 for c in comments for r in c.replies if r.resolved
        )

        render_comment_metrics(total + replies, resolved)

        st.session_state["_p1_comment_tab"] = st.session_state["p1_comment_tab"]
        comment_tab = st.pills(
            "View",
            ["Counts", "Timeline"],
            key="_p1_comment_tab",
            on_change=_store_comment_tab,
            selection_mode="single",
        )

        if comment_tab == "Counts":
            render_author_bar(filtered_c_df, "Count by Author")

        elif comment_tab == "Timeline":
            st.session_state["_p1_expanded_view"] = st.session_state["p1_expanded_view"]
            st.session_state["_p1_expand_all"] = st.session_state["p1_expand_all"]
            st.session_state["_p1_show_fields"] = st.session_state["p1_show_fields"]

            render_comment_timeline(
                filtered_c_df,
                "Timeline",
                expanded_view_key="_p1_expanded_view",
                expand_all_key="_p1_expand_all",
                show_fields_key="_p1_show_fields",
                on_expanded_view=_store_expanded_view,
                on_expand_all=_store_expand_all,
                on_show_fields=_store_show_fields,
            )

    # ---------------------------------------------------------------------------
    # Redlines
    # ---------------------------------------------------------------------------
    elif main_tab == "Redlines":
        if not filtered_r_df.empty:
            earliest = filtered_r_df["date"].min().strftime("%B %-d, %Y")
            end = reference_date.strftime("%B %-d, %Y")
            st.caption(f"From {earliest} → {end}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(filtered_r_df))
        col2.metric(
            "Insertions",
            int((filtered_r_df["kind"] == "insertion").sum())
            if not filtered_r_df.empty
            else 0,
        )
        col3.metric(
            "Deletions",
            int((filtered_r_df["kind"] == "deletion").sum())
            if not filtered_r_df.empty
            else 0,
        )

        render_author_bar(filtered_r_df, "Count by Author")

    # ---------------------------------------------------------------------------
    # Moves
    # ---------------------------------------------------------------------------
    elif main_tab == "Moves":
        if not filtered_m_df.empty:
            earliest = filtered_m_df["date"].min().strftime("%B %-d, %Y")
            end = reference_date.strftime("%B %-d, %Y")
            st.caption(f"From {earliest} → {end}")

        st.metric("Total Moves", len(filtered_m_df))
        render_author_bar(filtered_m_df, "Move Count by Author")
