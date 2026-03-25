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
    open: int


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
        open=total - resolved,
    )


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def latest_date(comments: list[Comment], redlines: list[Redline]) -> datetime | None:
    """Return the latest date across all comments and redlines."""
    dates = []
    for obj in [*comments, *redlines]:
        try:
            dates.append(datetime.fromisoformat(obj.date.rstrip("Z")))
        except ValueError:
            logger.warning("Skipping unparseable date %r", obj.date)
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
        try:
            dt = datetime.fromisoformat(c.date.rstrip("Z"))
            rows.append(
                {
                    "author": c.author,
                    "age_days": (now - dt).days,
                    "date": dt,
                    "resolved": c.resolved,
                    "kind": kind,
                    "comment": c.text,
                    "selected": c.context.selected_text if c.context else None,
                    "sentence": [s.text for s in c.context.sentences]
                    if c.context
                    else [],
                    "paragraph": c.context.paragraph_text if c.context else None,
                }
            )
        except ValueError:
            logger.warning("Skipping comment with unparseable date %r (id=%s)", c.date, c.id)

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
        try:
            dt = datetime.fromisoformat(r.date.rstrip("Z"))
            rows.append(
                {
                    "author": r.author,
                    "age_days": (now - dt).days,
                    "date": dt,
                    "kind": r.kind,
                    "text": r.text,
                    "sentence": [s.text for s in r.context.sentences]
                    if r.context
                    else [],
                    "paragraph": r.context.paragraph_text if r.context else None,
                }
            )
        except ValueError:
            logger.warning("Skipping redline with unparseable date %r (id=%s)", r.date, r.id)
    return pd.DataFrame(rows)


def move_ages_df(
    moves: list[Move],
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    now = reference_date or datetime.now()
    rows = []
    for m in moves:
        try:
            dt = datetime.fromisoformat(m.date.rstrip("Z"))
            rows.append(
                {
                    "author": m.author,
                    "age_days": (now - dt).days,
                    "date": dt,
                    "from_para_idx": m.from_para_idx,
                    "to_para_idx": m.to_para_idx,
                    "distance": abs(m.to_para_idx - m.from_para_idx),
                }
            )
        except ValueError:
            logger.warning("Skipping move with unparseable date %r (id=%s)", m.date, m.id)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def resolution_rate(comments: list[Comment]) -> float:
    if not comments:
        return 0.0
    df = pd.DataFrame([c.to_row() for c in comments])
    total = len(df)
    return float(df["resolved"].sum()) / total if total else 0.0


def thread_depth(comments: list[Comment]) -> pd.DataFrame:
    df = pd.DataFrame([c.to_row() for c in comments])
    mask = df["replies"] > 0
    threaded = df.loc[mask, ["id", "author", "text", "replies"]].copy()
    return threaded.sort_values("replies", ascending=False).reset_index(drop=True)


def paragraph_comment_density(
    comments: list[Comment], all_paragraphs: list[str]
) -> pd.DataFrame:
    comment_counts: dict[str, int] = {}
    resolved_counts: dict[str, int] = {}

    for c in comments:
        if c.context and c.context.paragraph_text:
            para = c.context.paragraph_text
            comment_counts[para] = comment_counts.get(para, 0) + 1
            resolved_counts[para] = resolved_counts.get(para, 0) + int(c.resolved)

    rows = []
    for i, para in enumerate(all_paragraphs):
        truncated = para[:80] + "…" if len(para) > 80 else para
        rows.append(
            {
                "index": i,
                "paragraph": truncated,
                "full": para,
                "comments": comment_counts.get(para, 0),
                "resolved": resolved_counts.get(para, 0),
                "unresolved": comment_counts.get(para, 0)
                - resolved_counts.get(para, 0),
            }
        )

    return pd.DataFrame(rows)


def time_bin(dates: list[datetime]) -> str:
    if len(dates) < 2:
        return "day"
    span = (max(dates) - min(dates)).days
    if span < 30:
        return "day"
    if span < 180:
        return "week"
    return "month"


# ---------------------------------------------------------------------------
# Comment summary
# ---------------------------------------------------------------------------
@dataclass
class CommentSummary:
    total: int
    resolved: int
    open: int
    resolution_rate: float
    total_authors: int
    earliest_comment: datetime | None
    latest_comment: datetime | None
    span_days: int | None
    avg_age_days: float | None
    avg_open_age_days: float | None


def comment_summary(
    comments: list[Comment],
    reference_date: datetime | None = None,
) -> CommentSummary:
    now = reference_date or datetime.now()
    dates: list[datetime] = []
    ages: list[int] = []
    open_ages: list[int] = []

    for c in comments:
        try:
            dt = datetime.fromisoformat(c.date.rstrip("Z"))
            dates.append(dt)
            age = (now - dt).days
            ages.append(age)
            if not c.resolved:
                open_ages.append(age)
        except ValueError:
            logger.warning("Skipping comment with unparseable date %r (id=%s)", c.date, c.id)

    resolved = sum(1 for c in comments if c.resolved)
    total = len(comments)

    return CommentSummary(
        total=total,
        resolved=resolved,
        open=total - resolved,
        resolution_rate=resolved / total if total else 0.0,
        total_authors=len({c.author for c in comments}),
        earliest_comment=min(dates) if dates else None,
        latest_comment=max(dates) if dates else None,
        span_days=(max(dates) - min(dates)).days if len(dates) > 1 else 0,
        avg_age_days=sum(ages) / len(ages) if ages else None,
        avg_open_age_days=sum(open_ages) / len(open_ages) if open_ages else None,
    )


# ---------------------------------------------------------------------------
# Redline summary
# ---------------------------------------------------------------------------
@dataclass
class RedlineSummary:
    total: int
    insertions: int
    deletions: int
    total_authors: int
    earliest_redline: datetime | None
    latest_redline: datetime | None
    span_days: int | None
    avg_age_days: float | None
    redlined_chars: int
    total_chars: int
    redline_density: float


def redline_summary(
    redlines: list[Redline],
    all_paragraphs: list[str],
    reference_date: datetime | None = None,
) -> RedlineSummary:
    redlined_chars = sum(len(r.text) for r in redlines)
    total_chars = sum(len(p) for p in all_paragraphs)

    if not redlines:
        return RedlineSummary(
            total=0,
            insertions=0,
            deletions=0,
            total_authors=0,
            earliest_redline=None,
            latest_redline=None,
            span_days=None,
            avg_age_days=None,
            redlined_chars=redlined_chars,
            total_chars=total_chars,
            redline_density=redlined_chars / total_chars if total_chars else 0.0,
        )

    now = reference_date or datetime.now()
    dates: list[datetime] = []
    ages: list[int] = []

    for r in redlines:
        try:
            dt = datetime.fromisoformat(r.date.rstrip("Z"))
            dates.append(dt)
            ages.append((now - dt).days)
        except ValueError:
            logger.warning("Skipping redline with unparseable date %r (id=%s)", r.date, r.id)

    return RedlineSummary(
        total=len(redlines),
        insertions=sum(1 for r in redlines if r.kind == "insertion"),
        deletions=sum(1 for r in redlines if r.kind == "deletion"),
        total_authors=len({r.author for r in redlines}),
        earliest_redline=min(dates) if dates else None,
        latest_redline=max(dates) if dates else None,
        span_days=(max(dates) - min(dates)).days if len(dates) > 1 else 0,
        avg_age_days=sum(ages) / len(ages) if ages else None,
        redlined_chars=redlined_chars,
        total_chars=total_chars,
        redline_density=redlined_chars / total_chars if total_chars else 0.0,
    )


# ---------------------------------------------------------------------------
# Problematic passages
# ---------------------------------------------------------------------------
@dataclass
class PassageActivity:
    para_idx: int
    paragraph: str
    comment_count: int
    redline_count: int
    insertion_count: int
    deletion_count: int
    move_count: int
    total_activity: int
    authors: set[str]


def problematic_passages(
    comments: list[Comment],
    redlines: list[Redline],
    moves: list[Move],
    all_paragraphs: list[str],
    min_activity: int = 1,
) -> list[PassageActivity]:
    """
    Rank paragraphs by total activity (comments + redlines + moves), descending.
    Moves are counted once at to_para_idx (the destination).
    """
    activity: dict[int, dict] = {
        i: {
            "comment_count": 0,
            "redline_count": 0,
            "insertion_count": 0,
            "deletion_count": 0,
            "move_count": 0,
            "authors": set(),
        }
        for i in range(len(all_paragraphs))
    }

    for c in comments:
        if c.context:
            for idx in range(c.context.start_para_idx, c.context.end_para_idx + 1):
                if idx in activity:
                    activity[idx]["comment_count"] += 1
                    activity[idx]["authors"].add(c.author)

    for r in redlines:
        if r.context:
            idx = r.context.para_idx
            if idx in activity:
                activity[idx]["redline_count"] += 1
                activity[idx]["authors"].add(r.author)
                if r.kind == "insertion":
                    activity[idx]["insertion_count"] += 1
                else:
                    activity[idx]["deletion_count"] += 1

    for m in moves:
        idx = m.to_para_idx
        if idx in activity:
            activity[idx]["move_count"] += 1
            activity[idx]["authors"].add(m.author)

    results = []
    for i, acc in activity.items():
        total = acc["comment_count"] + acc["redline_count"] + acc["move_count"]
        if total >= min_activity:
            results.append(
                PassageActivity(
                    para_idx=i,
                    paragraph=all_paragraphs[i],
                    comment_count=acc["comment_count"],
                    redline_count=acc["redline_count"],
                    insertion_count=acc["insertion_count"],
                    deletion_count=acc["deletion_count"],
                    move_count=acc["move_count"],
                    total_activity=total,
                    authors=acc["authors"],
                )
            )

    return sorted(results, key=lambda p: p.total_activity, reverse=True)
