"""
Issues Overview — multi-step NLI pipeline.

For each identified issue cluster:
  1. Score comment-thread contentiousness via NLI.
  2. Expand document context if the clause text is too short.
  3. Score the clause text against risk-language templates via NLI.
  4. Combine signals → risk level and party positions.

No external API calls — everything runs with the models already loaded by the app.
"""

import html
import io
from collections import defaultdict
from textwrap import shorten

from src.utils.page import require_document

import numpy as np
import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from src.app_state import KEY_ASSESSED_ISSUES
from src.comments.extract import Comment, extract_comments, extract_paragraphs
from src.nlp.entailment import contentiousness_score, template_score
from src.stats.render import AUTHOR_PALETTE
from src.utils.models import get_cross_encoder, get_sentence_transformer, get_spacy_nlp


# ---------------------------------------------------------------------------
# Risk templates  (NLI hypotheses evaluated against clause text)
# ---------------------------------------------------------------------------
_HIGH_TEMPLATES = [
    "The liability under this agreement is unlimited or uncapped.",
    "Consequential, indirect, or punitive damages are not excluded.",
    "This agreement can be terminated at any time without cause.",
    "Intellectual property rights are assigned permanently and irrevocably.",
    "The indemnification obligation covers all losses without limitation.",
    "There is no cap on the total amount of damages recoverable.",
]

_MEDIUM_TEMPLATES = [
    "There is a specified cap or ceiling on liability.",
    "Confidentiality obligations extend beyond the term of the agreement.",
    "Assignment of rights requires prior written consent.",
    "A defined notice period is required before termination.",
    "Exclusivity provisions restrict the party's ability to work with others.",
]

_LOW_TEMPLATES = [
    "This provision is a standard administrative or procedural matter.",
    "This clause provides a minor clarification with no material impact.",
    "This is a boilerplate definition with no financial consequence.",
]

_RISK_ORDER = {"high": 0, "medium": 1, "low": 2}
_RISK_COLOR = {"high": "#8b3535", "medium": "#7a5a20", "low": "#2e6b42"}
_RISK_LABEL = {"high": "High", "medium": "Medium", "low": "Low"}


_ACCEPTING_KW = {
    "agree",
    "agreed",
    "accept",
    "accepted",
    "approved",
    "approve",
    "ok",
    "fine",
    "no objection",
    "looks good",
    "confirmed",
}
_PUSHING_KW = {
    "propose",
    "suggested",
    "suggest",
    "reject",
    "rejected",
    "object",
    "disagree",
    "revise",
    "revised",
    "delete",
    "remove",
    "change",
    "modify",
    "alternative",
    "instead",
    "replace",
    "amend",
    "redline",
}


# ---------------------------------------------------------------------------
# Comment helpers
# ---------------------------------------------------------------------------
def _thread_text(c: Comment) -> str:
    """Concatenate selected text, comment body, and all replies for embedding."""
    parts = []
    if c.context and c.context.selected_text:
        parts.append(c.context.selected_text)
    if c.text.strip():
        parts.append(c.text)
    for r in c.replies:
        if r.text.strip():
            parts.append(r.text)
    return " ".join(parts)


def _primary_clause(comments: list[Comment]) -> tuple[str, int]:
    """Return (clause_text, para_idx) from the comment with the longest selected text."""
    best: Comment | None = None
    best_len = -1
    for c in comments:
        length = len((c.context.selected_text or "") if c.context else "")
        if length > best_len:
            best_len = length
            best = c
    if best and best.context and best.context.selected_text:
        return best.context.selected_text, best.context.start_para_idx
    # Fallback: first comment's text and a para_idx of 0
    return (comments[0].text if comments else ""), 0


def _expand_context(paragraphs: list[str], para_idx: int, window: int = 3) -> str:
    start = max(0, para_idx - window)
    end = min(len(paragraphs) - 1, para_idx + window)
    return " ".join(
        (paragraphs[i] or "").strip()
        for i in range(start, end + 1)
        if (paragraphs[i] or "").strip()
    )


def _issue_name(comments: list[Comment], centroid_idx: int, nlp) -> str:
    """Derive a short issue label using spaCy noun chunks."""
    c = comments[centroid_idx]
    source = (c.context.selected_text if c.context else "") or c.text
    if not source:
        return "Unnamed Issue"
    doc = nlp(source[:300])
    chunks = sorted(
        [ch.text.strip() for ch in doc.noun_chunks if len(ch.text.strip()) > 3],
        key=len,
        reverse=True,
    )
    if chunks:
        return " / ".join(chunks[:2])
    return shorten(source, 60, placeholder="…")


def _party_stances(comments: list[Comment]) -> list[dict]:
    """Keyword-based stance detection per author."""
    by_author: dict[str, list[str]] = defaultdict(list)
    for c in comments:
        if c.text.strip():
            by_author[c.author].append(c.text)
        for r in c.replies:
            if r.text.strip():
                by_author[r.author].append(r.text)

    positions = []
    for author, texts in by_author.items():
        combined_lower = " ".join(texts).lower()
        is_accepting = any(kw in combined_lower for kw in _ACCEPTING_KW)
        is_pushing = any(kw in combined_lower for kw in _PUSHING_KW)

        if is_pushing and not is_accepting:
            stance = _best_snippet(texts, "proposes changes")
        elif is_accepting and not is_pushing:
            stance = "accepts current language"
        else:
            stance = _best_snippet(texts, "position unclear")

        positions.append({"party": author, "stance": stance})
    return positions


def _best_snippet(texts: list[str], default: str) -> str:
    for t in texts:
        t = t.strip()
        if len(t) > 20:
            return shorten(t, 120, placeholder="…")
    return default


# ---------------------------------------------------------------------------
# Issue assessor
# ---------------------------------------------------------------------------
def _assess_issue(
    group: list[Comment],
    paragraphs: list[str],
    nli: CrossEncoder,
    nlp,
    all_embeddings: np.ndarray,
    group_indices: list[int],
) -> dict:
    """
    Multi-step NLI assessment for one issue cluster.

    1. Get initial clause text.
    2. Score contentiousness (NLI).
    3. Expand document context if clause is too short.
    4. Score clause against risk templates (NLI).
    5. Combine → risk level, positions, rationale.
    """
    # 1. Clause text + location
    clause_text, para_idx = _primary_clause(group)

    # 2. Contentiousness
    comment_texts = [c.text for c in group] + [r.text for c in group for r in c.replies]
    cont = contentiousness_score(nli, comment_texts)

    # 3. Expand context if clause is too short or contentiousness is ambiguous.
    short_clause = len(clause_text.strip()) < 80
    ambiguous_cont = 0.15 < cont < 0.55
    context_expanded = False

    if short_clause or ambiguous_cont:
        scoring_text = _expand_context(paragraphs, para_idx, window=3)
        context_expanded = bool(scoring_text.strip())
    else:
        scoring_text = clause_text

    if not scoring_text.strip():
        scoring_text = clause_text  # last resort

    # 4. Template scoring
    high_score = template_score(nli, scoring_text, _HIGH_TEMPLATES)
    med_score = template_score(nli, scoring_text, _MEDIUM_TEMPLATES)

    # 5. Risk level: template signal + contentiousness heat
    effective_high = high_score * 0.6 + (cont if cont > 0.4 else 0.0) * 0.4
    effective_med = med_score * 0.6 + (cont if 0.2 < cont <= 0.4 else 0.0) * 0.4

    if effective_high > 0.35 or (high_score > 0.25 and cont > 0.35):
        risk_level = "high"
    elif effective_med > 0.3 or cont > 0.25 or med_score > 0.3:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Centroid comment for naming
    cluster_embs = all_embeddings[group_indices]
    centroid = cluster_embs.mean(axis=0)
    centroid_local = int(np.linalg.norm(cluster_embs - centroid, axis=1).argmin())

    rationale_parts = []
    if cont > 0.3:
        rationale_parts.append(
            f"active disagreement between commenters (score {cont:.2f})"
        )
    if high_score > 0.25:
        rationale_parts.append(
            f"clause matches high-risk language patterns (score {high_score:.2f})"
        )
    if med_score > 0.3 and risk_level == "medium":
        rationale_parts.append(
            f"clause matches standard risk patterns (score {med_score:.2f})"
        )
    if not rationale_parts:
        rationale_parts.append("routine clause with low negotiation friction")
    if context_expanded:
        rationale_parts.append("context expanded to surrounding paragraphs")

    open_comments = [c for c in group if not c.resolved]

    return {
        "issue_name": _issue_name(group, centroid_local, nlp),
        "risk_level": risk_level,
        "risk_rationale": "; ".join(rationale_parts).capitalize() + ".",
        "clause_text": clause_text,
        "para_idx": para_idx,
        "contentiousness": cont,
        "high_template_score": high_score,
        "status": "open" if open_comments else "resolved",
        "positions": _party_stances(group),
        "n_comments": len(group),
        "context_expanded": context_expanded,
    }


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _embed_threads(texts: tuple[str, ...]) -> np.ndarray:
    return get_sentence_transformer("all-MiniLM-L6-v2").encode(
        list(texts), show_progress_bar=False, normalize_embeddings=True
    )


def _cluster_comments(comments: list[Comment], n: int) -> list[list[int]]:
    texts = tuple(_thread_text(c) for c in comments)
    embeddings = _embed_threads(texts)
    n_clusters = min(n, len(comments))
    if n_clusters <= 1:
        return [list(range(len(comments)))]
    labels = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    ).fit_predict(embeddings)
    groups: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        groups[int(label)].append(i)
    return list(groups.values())


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------
def _initials(name: str) -> str:
    parts = name.strip().split()
    return "".join(p[0].upper() for p in parts[:2]) if parts else "?"


def _positions_html(positions: list[dict], color_map: dict[str, str]) -> str:
    if not positions:
        return ""
    rows = []
    for pos in positions:
        party = pos.get("party", "Unknown")
        stance = html.escape(pos.get("stance", ""))
        bg_color = color_map.get(party, AUTHOR_PALETTE[0])
        initials = _initials(party)
        avatar = (
            f'<div style="width:28px;height:28px;border-radius:50%;background:{bg_color};'
            f"display:flex;align-items:center;justify-content:center;"
            f'font-size:0.65em;font-weight:700;flex-shrink:0;color:#333">{initials}</div>'
        )
        body = (
            f'<div style="flex:1;min-width:0">'
            f'<div style="font-weight:600;font-size:0.84em;color:#333;margin-bottom:1px">{html.escape(party)}</div>'
            f'<div style="font-size:0.82em;color:#666;line-height:1.45">{stance}</div>'
            f"</div>"
        )
        rows.append(
            f'<div style="display:flex;align-items:flex-start;gap:10px;margin:7px 0">{avatar}{body}</div>'
        )
    return "".join(rows)


def _issue_card(issue: dict, color_map: dict[str, str]) -> str:
    risk = issue["risk_level"]
    color = _RISK_COLOR[risk]
    label = _RISK_LABEL[risk]

    is_open = issue["status"] == "open"
    status_label = "Open" if is_open else "Resolved"
    status_color = "#888"

    name_h = html.escape(issue["issue_name"])
    rationale_h = html.escape(issue.get("risk_rationale", ""))

    clause = (issue.get("clause_text") or "").strip()
    clause_block = ""
    if clause:
        clause_h = html.escape(shorten(clause, 220, placeholder="…"))
        clause_block = (
            f'<div style="margin:0 20px 14px 20px;background:#f9f9f9;border-left:3px solid {color};'
            f'padding:9px 13px;font-size:0.86em;color:#555;line-height:1.55;font-style:italic">'
            f"{clause_h}</div>"
        )

    positions = _positions_html(issue.get("positions", []), color_map)
    positions_block = ""
    if positions:
        positions_block = (
            f'<div style="border-top:1px solid #f0f0f0"></div>'
            f'<div style="padding:10px 20px 12px 20px">{positions}</div>'
        )

    footer_parts = [
        f"¶{issue['para_idx']}",
        f"{issue['n_comments']} comment{'s' if issue['n_comments'] != 1 else ''}",
    ]
    footer_inner = "&ensp;·&ensp;".join(footer_parts)

    badge = (
        f'<span style="font-size:0.75em;font-weight:600;color:{color};'
        f'border:1px solid {color};padding:2px 9px;border-radius:10px">{label}</span>'
    )
    status_span = (
        f'<span style="font-size:0.78em;color:{status_color}">{status_label}</span>'
    )
    header = (
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;'
        f'padding:15px 20px 11px 20px">'
        f'<div style="font-weight:600;font-size:0.97em;color:#1a1a2e;line-height:1.35;margin-right:14px">{name_h}</div>'
        f'<div style="display:flex;gap:8px;align-items:center;flex-shrink:0;margin-top:2px">{badge}{status_span}</div>'
        f"</div>"
    )
    rationale_row = (
        f'<div style="padding:0 20px 13px 20px;font-size:0.8em;color:#777;line-height:1.45">'
        f"{rationale_h}</div>"
    )
    footer_row = (
        f'<div style="background:#fafafa;border-top:1px solid #f0f0f0;padding:6px 20px;'
        f'font-size:0.74em;color:#bbb">{footer_inner}</div>'
    )

    return (
        f'<div style="border-radius:6px;border:1px solid #e8eaed;border-top:3px solid {color};'
        f'background:#fff;margin-bottom:14px;overflow:hidden">'
        f"{header}{clause_block}{rationale_row}{positions_block}{footer_row}"
        f"</div>"
    )


def _summary_row(high_n: int, med_n: int, low_n: int, total_n: int) -> str:
    items = [
        (str(total_n), "issues", "#262730"),
        (str(high_n), "high", _RISK_COLOR["high"]),
        (str(med_n), "medium", _RISK_COLOR["medium"]),
        (str(low_n), "low", _RISK_COLOR["low"]),
    ]
    cells = "".join(
        f'<div style="flex:1;text-align:center;padding:12px 0;border-right:1px solid #f0f0f0">'
        f'<div style="font-size:1.7em;font-weight:700;color:#262730;line-height:1">{v}</div>'
        f'<div style="font-size:0.72em;color:{c};margin-top:3px;text-transform:uppercase;letter-spacing:0.06em">{l}</div>'
        f"</div>"
        for v, l, c in items
    )
    cells = cells.rsplit("border-right:1px solid #f0f0f0", 1)
    cells = "border-right:none".join(cells)
    return (
        f'<div style="display:flex;border:1px solid #e8eaed;border-radius:6px;'
        f'background:#fff;margin-bottom:16px">{cells}</div>'
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_data(file_bytes: bytes):
    comments, _ = extract_comments(io.BytesIO(file_bytes))
    doc = extract_paragraphs(io.BytesIO(file_bytes))
    return comments, doc.paragraphs


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
file_bytes = require_document()

all_comments, paragraphs = _load_data(file_bytes)
top_level = [c for c in all_comments if not c.parent_id]

# Sidebar
st.sidebar.markdown("### Settings")
nli_model_name = st.sidebar.selectbox(
    "NLI model",
    ["cross-encoder/nli-deberta-v3-small", "cross-encoder/nli-deberta-v3-base"],
    index=0,
    help="Small is faster; base is more accurate.",
)
open_only = st.sidebar.checkbox("Open issues only", value=True)

max_issues = max(2, len(top_level))
n_issues = st.sidebar.slider(
    "Number of issues",
    min_value=2,
    max_value=min(20, max_issues),
    value=min(6, max_issues // 2 + 1),
)

# Filter
working = [c for c in top_level if not c.resolved] if open_only else top_level

# Main
st.subheader("Issues Overview")
if not working:
    st.warning(
        "No open comments found. Uncheck **Open issues only** in the sidebar to include resolved ones."
    )
    st.stop()

open_count = sum(1 for c in top_level if not c.resolved)
resolved_count = len(top_level) - open_count
st.caption(
    f"{len(top_level)} comment threads · {open_count} open · {resolved_count} resolved · analysing {len(working)}"
)

if st.button("Generate Report", type="primary"):
    st.session_state.pop(KEY_ASSESSED_ISSUES, None)

    with st.spinner("Loading models…"):
        nli = get_cross_encoder(nli_model_name)
        nlp = get_spacy_nlp()

    texts = tuple(_thread_text(c) for c in working)
    embeddings = _embed_threads(texts)
    groups = _cluster_comments(working, n_issues)

    issues: list[dict] = []
    prog = st.progress(0.0, text="Analyzing issues…")
    status_line = st.empty()

    for gi, group_indices in enumerate(groups):
        group_comments = [working[i] for i in group_indices]
        preview = shorten(group_comments[0].text or "", 60, placeholder="…")
        status_line.caption(f"Issue {gi + 1}/{len(groups)}: \u201c{preview}\u201d")

        issue = _assess_issue(
            group_comments, paragraphs, nli, nlp, embeddings, group_indices
        )
        issues.append(issue)
        prog.progress((gi + 1) / len(groups))

    status_line.empty()
    prog.empty()
    st.session_state[KEY_ASSESSED_ISSUES] = issues

# ---------------------------------------------------------------------------
# Render results
# ---------------------------------------------------------------------------
issues: list[dict] = st.session_state.get(KEY_ASSESSED_ISSUES, [])
if not issues:
    st.stop()

sorted_issues = sorted(issues, key=lambda x: _RISK_ORDER.get(x["risk_level"], 2))

high_n = sum(1 for i in sorted_issues if i["risk_level"] == "high")
med_n = sum(1 for i in sorted_issues if i["risk_level"] == "medium")
low_n = sum(1 for i in sorted_issues if i["risk_level"] == "low")
total_n = len(sorted_issues)

# Summary row
st.markdown(_summary_row(high_n, med_n, low_n, total_n), unsafe_allow_html=True)


# Build global author color map (sorted, matching document_statistics assignment)
all_authors = sorted(
    {pos["party"] for issue in sorted_issues for pos in issue.get("positions", [])}
)
author_color_map = {
    author: AUTHOR_PALETTE[i % len(AUTHOR_PALETTE)]
    for i, author in enumerate(all_authors)
}

# Issue cards
for issue in sorted_issues:
    st.markdown(_issue_card(issue, author_color_map), unsafe_allow_html=True)
