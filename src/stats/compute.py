from src.comments.extract import Comment
from datetime import datetime
import pandas as pd
from dataclasses import dataclass
from src.redlines.extract import Redline


def resolution_rate(comments: list[Comment]) -> float:
    if not comments:
        return 0.0
    df = pd.DataFrame([c.to_row() for c in comments])
    total = len(df)
    resolved = df["resolved"].sum()
    return resolved / total if total else 0.0


def thread_depth(comments: list[Comment]) -> pd.DataFrame:
    df = pd.DataFrame([c.to_row() for c in comments])
    return (
        df[df["replies"] > 0][["id", "author", "text", "replies"]]
        .sort_values("replies", ascending=False)
        .reset_index(drop=True)
    )


def open_comment_ages(
    comments: list[Comment],
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    now = reference_date or datetime.now()
    for c in comments:
        if not c.resolved:
            try:
                dt = datetime.fromisoformat(c.date.rstrip("Z"))
                age = (now - dt).days
                rows.append({**c.to_row(), "age_days": age})
            except ValueError:
                pass
    return pd.DataFrame(rows)


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


@dataclass
class CommentSummary:
    total: int
    resolved: int
    open: int
    resolution_rate: float
    total_authors: int
    earliest_comment: datetime | None
    latest_comment: datetime | None
    span_days: int | None  # days between earliest and latest
    avg_age_days: float | None  # average age of all comments
    avg_open_age_days: float | None  # average age of open comments only


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
            pass

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
    redline_density: float  # redlined_chars / total_chars


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
            pass

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
