import logging
from dataclasses import dataclass, field
from datetime import datetime, date

import pandas as pd

logger = logging.getLogger(__name__)

from src.comments.extract import Comment
from src.redlines.extract import Redline, Move


# ---------------------------------------------------------------------------
# Document data container
# ---------------------------------------------------------------------------
@dataclass
class DocumentData:
    """
    All raw extracted data from a single .docx file.
    Intended as the single source of truth passed around the app.
    """

    comments: list[Comment]
    redlines: list[Redline]
    moves: list[Move]
    all_paragraphs: list[str]


# ---------------------------------------------------------------------------
# Comment metrics
# ---------------------------------------------------------------------------
@dataclass
class CommentMetrics:
    """Counts derived from a comment list, including replies."""

    total: int
    top_level: int
    replies: int
    resolved: int


def comment_metrics(comments: list[Comment]) -> CommentMetrics:
    top_level = len(comments)
    replies = sum(len(c.replies) for c in comments)
    resolved = sum(1 for c in comments if c.resolved) + sum(
        1 for c in comments for r in c.replies if r.resolved
    )
    total = top_level + replies
    return CommentMetrics(
        total=total,
        top_level=top_level,
        replies=replies,
        resolved=resolved,
    )


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def _parse_dt(obj, kind: str) -> datetime | None:
    """Parse obj.date as ISO 8601, logging a warning and returning None on failure."""
    try:
        return datetime.fromisoformat(obj.date.rstrip("Z"))
    except ValueError:
        logger.warning(
            "Skipping %s with unparseable date %r (id=%s)", kind, obj.date, obj.id
        )
        return None


def latest_date(comments: list[Comment], redlines: list[Redline]) -> datetime | None:
    """Return the latest date across all comments and redlines."""
    dates = [
        dt
        for obj, kind in [
            *[(c, "comment") for c in comments],
            *[(r, "redline") for r in redlines],
        ]
        if (dt := _parse_dt(obj, kind)) is not None
    ]
    return max(dates) if dates else None


def filter_by_date(df: pd.DataFrame, date_min: date, date_max: date) -> pd.DataFrame:
    """Filter a DataFrame with a 'date' column to the given date range."""
    if df.empty:
        return df
    return df[
        (df["date"].dt.date >= date_min) & (df["date"].dt.date <= date_max)
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Age DataFrames
# ---------------------------------------------------------------------------
def comment_ages_df(
    comments: list[Comment],
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    now = reference_date or datetime.now()
    rows = []

    def _add(c: Comment, kind: str) -> None:
        dt = _parse_dt(c, kind)
        if dt is None:
            return
        rows.append(
            {
                "author": c.author,
                "age_days": (now - dt).days,
                "date": dt,
                "resolved": c.resolved,
                "kind": kind,
                "comment": c.text,
                "selected": c.context.selected_text if c.context else None,
                "sentence": [s.text for s in c.context.sentences] if c.context else [],
                "paragraph": c.context.paragraph_text if c.context else None,
            }
        )

    for c in comments:
        _add(c, "comment")
        for reply in c.replies:
            _add(reply, "reply")

    return pd.DataFrame(rows)


def redline_ages_df(
    redlines: list[Redline],
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    now = reference_date or datetime.now()
    rows = []
    for r in redlines:
        dt = _parse_dt(r, "redline")
        if dt is None:
            continue
        rows.append(
            {
                "author": r.author,
                "age_days": (now - dt).days,
                "date": dt,
                "kind": r.kind,
                "text": r.text,
                "sentence": [s.text for s in r.context.sentences] if r.context else [],
                "paragraph": r.context.paragraph_text if r.context else None,
            }
        )
    return pd.DataFrame(rows)


def move_ages_df(
    moves: list[Move],
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    now = reference_date or datetime.now()
    rows = []
    for m in moves:
        dt = _parse_dt(m, "move")
        if dt is None:
            continue
        rows.append(
            {
                "author": m.author,
                "age_days": (now - dt).days,
                "date": dt,
                "text": m.text,
                "from_para_idx": m.from_para_idx,
                "to_para_idx": m.to_para_idx,
                "distance": abs(m.to_para_idx - m.from_para_idx),
            }
        )
    return pd.DataFrame(rows)
