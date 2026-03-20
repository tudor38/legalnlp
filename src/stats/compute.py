from src.comments.extract import Comment
from datetime import datetime
import pandas as pd


def resolution_rate(comments: list[Comment]) -> float:
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


def open_comment_ages(comments: list[Comment]) -> pd.DataFrame:
    now = datetime.now()
    rows = []
    for c in comments:
        if not c.resolved:
            try:
                dt = datetime.fromisoformat(c.date.rstrip("Z"))
                age = (now - dt).days
                rows.append({**c.to_row(), "age_days": age})
            except ValueError:
                pass
    return pd.DataFrame(rows)
