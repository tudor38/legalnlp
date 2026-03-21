from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.comments.extract import Comment
from src.redlines.extract import Redline, Move


def resolution_rate(comments: list[Comment]) -> float:
    if not comments:
        return 0.0
    df = pd.DataFrame([c.to_row() for c in comments])
    total    = len(df)
    resolved = df["resolved"].sum()
    return resolved / total if total else 0.0


def thread_depth(comments: list[Comment]) -> pd.DataFrame:
    df       = pd.DataFrame([c.to_row() for c in comments])
    mask     = df["replies"] > 0
    threaded = df.loc[mask, ["id", "author", "text", "replies"]].copy()
    return threaded.sort_values("replies", ascending=False).reset_index(drop=True)


def open_comment_ages(
    comments: list[Comment],
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    now  = reference_date or datetime.now()
    rows = []
    for c in comments:
        if not c.resolved:
            try:
                dt  = datetime.fromisoformat(c.date.rstrip("Z"))
                age = (now - dt).days
                rows.append({**c.to_row(), "age_days": age})
            except ValueError:
                pass
    return pd.DataFrame(rows)


def paragraph_comment_density(
    comments: list[Comment], all_paragraphs: list[str]
) -> pd.DataFrame:
    comment_counts:  dict[str, int] = {}
    resolved_counts: dict[str, int] = {}

    for c in comments:
        if c.context and c.context.paragraph_text:
            para = c.context.paragraph_text
            comment_counts[para]  = comment_counts.get(para, 0) + 1
            resolved_counts[para] = resolved_counts.get(para, 0) + int(c.resolved)

    rows = []
    for i, para in enumerate(all_paragraphs):
        truncated = para[:80] + "…" if len(para) > 80 else para
        rows.append({
            "index":      i,
            "paragraph":  truncated,
            "full":       para,
            "comments":   comment_counts.get(para, 0),
            "resolved":   resolved_counts.get(para, 0),
            "unresolved": comment_counts.get(para, 0) - resolved_counts.get(para, 0),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comment summary
# ---------------------------------------------------------------------------
@dataclass
class CommentSummary:
    total:             int
    resolved:          int
    open:              int
    resolution_rate:   float
    total_authors:     int
    earliest_comment:  datetime | None
    latest_comment:    datetime | None
    span_days:         int | None
    avg_age_days:      float | None
    avg_open_age_days: float | None


def comment_summary(
    comments: list[Comment],
    reference_date: datetime | None = None,
) -> CommentSummary:
    now        = reference_date or datetime.now()
    dates:     list[datetime] = []
    ages:      list[int]      = []
    open_ages: list[int]      = []

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
    total    = len(comments)

    return CommentSummary(
        total             = total,
        resolved          = resolved,
        open              = total - resolved,
        resolution_rate   = resolved / total if total else 0.0,
        total_authors     = len({c.author for c in comments}),
        earliest_comment  = min(dates) if dates else None,
        latest_comment    = max(dates) if dates else None,
        span_days         = (max(dates) - min(dates)).days if len(dates) > 1 else 0,
        avg_age_days      = sum(ages) / len(ages) if ages else None,
        avg_open_age_days = sum(open_ages) / len(open_ages) if open_ages else None,
    )


# ---------------------------------------------------------------------------
# Redline summary
# ---------------------------------------------------------------------------
@dataclass
class RedlineSummary:
    total:            int
    insertions:       int
    deletions:        int
    total_authors:    int
    earliest_redline: datetime | None
    latest_redline:   datetime | None
    span_days:        int | None
    avg_age_days:     float | None
    redlined_chars:   int
    total_chars:      int
    redline_density:  float


def redline_summary(
    redlines:       list[Redline],
    all_paragraphs: list[str],
    reference_date: datetime | None = None,
) -> RedlineSummary:
    redlined_chars = sum(len(r.text) for r in redlines)
    total_chars    = sum(len(p) for p in all_paragraphs)

    if not redlines:
        return RedlineSummary(
            total            = 0,
            insertions       = 0,
            deletions        = 0,
            total_authors    = 0,
            earliest_redline = None,
            latest_redline   = None,
            span_days        = None,
            avg_age_days     = None,
            redlined_chars   = redlined_chars,
            total_chars      = total_chars,
            redline_density  = redlined_chars / total_chars if total_chars else 0.0,
        )

    now   = reference_date or datetime.now()
    dates: list[datetime] = []
    ages:  list[int]      = []

    for r in redlines:
        try:
            dt = datetime.fromisoformat(r.date.rstrip("Z"))
            dates.append(dt)
            ages.append((now - dt).days)
        except ValueError:
            pass

    return RedlineSummary(
        total            = len(redlines),
        insertions       = sum(1 for r in redlines if r.kind == "insertion"),
        deletions        = sum(1 for r in redlines if r.kind == "deletion"),
        total_authors    = len({r.author for r in redlines}),
        earliest_redline = min(dates) if dates else None,
        latest_redline   = max(dates) if dates else None,
        span_days        = (max(dates) - min(dates)).days if len(dates) > 1 else 0,
        avg_age_days     = sum(ages) / len(ages) if ages else None,
        redlined_chars   = redlined_chars,
        total_chars      = total_chars,
        redline_density  = redlined_chars / total_chars if total_chars else 0.0,
    )


# ---------------------------------------------------------------------------
# Problematic passages
# ---------------------------------------------------------------------------
@dataclass
class PassageActivity:
    para_idx:        int
    paragraph:       str
    comment_count:   int
    redline_count:   int
    insertion_count: int
    deletion_count:  int
    move_count:      int       # moves landing at this paragraph (to_para_idx)
    total_activity:  int       # comment_count + redline_count + move_count
    authors:         set[str]


def problematic_passages(
    comments:       list[Comment],
    redlines:       list[Redline],
    moves:          list[Move],
    all_paragraphs: list[str],
    min_activity:   int = 1,
) -> list[PassageActivity]:
    """
    Rank paragraphs by total activity (comments + redlines + moves), descending.

    Moves are counted once at to_para_idx (the destination), since that is
    where the content lives in the final document.

    Parameters
    ----------
    min_activity : only return paragraphs with at least this many combined
                   events. Default 1 excludes untouched paragraphs.
    """
    activity: dict[int, dict] = {
        i: {
            "comment_count":   0,
            "redline_count":   0,
            "insertion_count": 0,
            "deletion_count":  0,
            "move_count":      0,
            "authors":         set(),
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
                    activity[idx]["deletion_count"]  += 1

    for m in moves:
        idx = m.to_para_idx   # count at destination
        if idx in activity:
            activity[idx]["move_count"] += 1
            activity[idx]["authors"].add(m.author)

    results = []
    for i, acc in activity.items():
        total = acc["comment_count"] + acc["redline_count"] + acc["move_count"]
        if total >= min_activity:
            results.append(PassageActivity(
                para_idx        = i,
                paragraph       = all_paragraphs[i],
                comment_count   = acc["comment_count"],
                redline_count   = acc["redline_count"],
                insertion_count = acc["insertion_count"],
                deletion_count  = acc["deletion_count"],
                move_count      = acc["move_count"],
                total_activity  = total,
                authors         = acc["authors"],
            ))

    return sorted(results, key=lambda p: p.total_activity, reverse=True)
