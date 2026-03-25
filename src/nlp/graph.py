"""
Entity and relationship extraction for legal contracts.

Produces a serialized property graph (nodes with metadata, typed directed
edges with provenance) that can be visualized as a Plotly network figure.
"""
from __future__ import annotations

import re
from collections import defaultdict

import networkx as nx
import streamlit as st

from src.utils.models import get_spacy_nlp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PARTY_NER = {"ORG", "PERSON"}
_CONTEXT_NER = {"GPE", "DATE", "MONEY"}
_ALL_NER = _PARTY_NER | _CONTEXT_NER

_NER_TO_NODE_TYPE = {
    "ORG": "PARTY",
    "PERSON": "PARTY",
    "GPE": "GPE",
    "DATE": "DATE",
    "MONEY": "MONEY",
}

# Maps verb lemmas → relation type
_VERB_MAP: dict[str, str] = {
    # Obligations
    "agree": "OBLIGATED_TO",
    "assign": "OBLIGATED_TO",
    "certify": "OBLIGATED_TO",
    "commit": "OBLIGATED_TO",
    "covenant": "OBLIGATED_TO",
    "defend": "OBLIGATED_TO",
    "deliver": "OBLIGATED_TO",
    "ensure": "OBLIGATED_TO",
    "guarantee": "OBLIGATED_TO",
    "hold": "OBLIGATED_TO",
    "implement": "OBLIGATED_TO",
    "maintain": "OBLIGATED_TO",
    "must": "OBLIGATED_TO",
    "notify": "OBLIGATED_TO",
    "obligate": "OBLIGATED_TO",
    "perform": "OBLIGATED_TO",
    "protect": "OBLIGATED_TO",
    "provide": "OBLIGATED_TO",
    "remedy": "OBLIGATED_TO",
    "represent": "OBLIGATED_TO",
    "return": "OBLIGATED_TO",
    "shall": "OBLIGATED_TO",
    "submit": "OBLIGATED_TO",
    "transfer": "OBLIGATED_TO",
    "undertake": "OBLIGATED_TO",
    "warrant": "OBLIGATED_TO",
    # Permissions
    "allow": "PERMITS",
    "authorize": "PERMITS",
    "grant": "PERMITS",
    "license": "PERMITS",
    "licence": "PERMITS",
    "may": "PERMITS",
    "permit": "PERMITS",
    "use": "PERMITS",
    # Restrictions
    "forbid": "RESTRICTS",
    "limit": "RESTRICTS",
    "preclude": "RESTRICTS",
    "prevent": "RESTRICTS",
    "prohibit": "RESTRICTS",
    "restrict": "RESTRICTS",
    # Governed by
    "apply": "GOVERNED_BY",
    "control": "GOVERNED_BY",
    "govern": "GOVERNED_BY",
    # Payment
    "compensate": "PAYS_TO",
    "indemnify": "PAYS_TO",
    "pay": "PAYS_TO",
    "reimburse": "PAYS_TO",
    # Contracted with
    "conclude": "CONTRACTED_WITH",
    "enter": "CONTRACTED_WITH",
    "execute": "CONTRACTED_WITH",
    "sign": "CONTRACTED_WITH",
}

_NEG_FLIP: dict[str, str] = {
    "OBLIGATED_TO": "RESTRICTS",
    "PERMITS": "RESTRICTS",
}

# Regex patterns for relationship extraction (idiomatic contract constructs)
_STRIP_PREFIX = re.compile(r"^(the|a|an|such|each|either|both)\s+", re.IGNORECASE)
_MULTI_SPACE = re.compile(r"\s{2,}")
_POSSESSIVE = re.compile(r"['\u2019]s?\s*$")

_DEFINITION_PATTERNS = [
    re.compile(
        r'["\u201c\u2018](?P<term>[^"\u201d\u2019]{1,80})["\u201d\u2019]\s+(?:shall\s+)?(?:mean|have\s+the\s+meaning)',
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<term>[A-Z][A-Za-z\s]{1,60})\b\s+(?:is\s+)?(?:defined\s+as|referred\s+to\s+(?:herein\s+)?as)",
        re.IGNORECASE,
    ),
    re.compile(
        r'\((?:hereinafter\s+)?(?:the\s+)?["\u201c\u2018](?P<term>[^"\u201d\u2019]{1,80})["\u201d\u2019]\)',
        re.IGNORECASE,
    ),
]

# (compiled_regex, relation_type, match_mode)
# match_mode "ab" = has group a and b; "b_only" = only dst group
_PARTY_WORD = r"[A-Z][A-Za-z\s&]{2,50}?"
_END = r"(?=[,;.()\'\u2019\s]|$)"

_REGEX_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # "between X and Y" → CONTRACTED_WITH
    (
        re.compile(
            rf"\bbetween\s+(?P<a>{_PARTY_WORD})\s+and\s+(?P<b>[A-Z][^,\n.;]{{2,50?}}?)"
            r"(?=\s*[,;.(]|\s+(?:dated|effective|for|regarding|\())",
            re.IGNORECASE,
        ),
        "CONTRACTED_WITH",
        "ab",
    ),
    # "X and Y hereby agree / enter into" → CONTRACTED_WITH
    (
        re.compile(
            rf"\b(?P<a>{_PARTY_WORD})\s+and\s+(?P<b>{_PARTY_WORD})\s+"
            r"(?:hereby\s+)?(?:agree|enter\s+into)\b",
            re.IGNORECASE,
        ),
        "CONTRACTED_WITH",
        "ab",
    ),
    # "governed by [the laws of] X" → GOVERNED_BY
    (
        re.compile(
            r"\bgoverned\s+by\s+(?:the\s+)?(?:laws?\s+of\s+(?:the\s+)?)?(?P<b>[A-Z][a-zA-Z\s]{2,40}?)(?:\s+law)?(?=[,;.]|\s*$)",
            re.IGNORECASE,
        ),
        "GOVERNED_BY",
        "b_only",
    ),
    # "X shall pay [to] Y" → PAYS_TO
    (
        re.compile(
            rf"\b(?P<a>{_PARTY_WORD})\s+shall\s+pay\s+(?:to\s+)?(?P<b>{_PARTY_WORD}){_END}",
            re.IGNORECASE,
        ),
        "PAYS_TO",
        "ab",
    ),
    # "X shall [verb] Y" → OBLIGATED_TO  (generic shall-obligation)
    (
        re.compile(
            rf"\b(?P<a>{_PARTY_WORD})\s+shall\s+(?:not\s+)?(?:be\s+)?(?:[a-z]{{2,15}}\s+){{0,2}}(?P<b>{_PARTY_WORD}){_END}",
            re.IGNORECASE,
        ),
        "OBLIGATED_TO",
        "ab",
    ),
]

# ---------------------------------------------------------------------------
# Visualization config (exported for use in the page)
# ---------------------------------------------------------------------------

NODE_COLORS: dict[str, str] = {
    "PARTY": "#4C72B0",
    "GPE": "#55A868",
    "DATE": "#DD8452",
    "MONEY": "#8172B3",
    "DEFINED_TERM": "#937860",
}

RELATION_COLORS: dict[str, str] = {
    "CONTRACTED_WITH": "#264653",
    "OBLIGATED_TO": "#e63946",
    "PERMITS": "#2a9d8f",
    "RESTRICTS": "#e76f51",
    "GOVERNED_BY": "#457b9d",
    "PAYS_TO": "#8172B3",
    "DEFINED_AS": "#937860",
    "GENERIC": "#aaaaaa",
}

ALL_RELATION_TYPES = list(RELATION_COLORS.keys())
ALL_NODE_TYPES = list(NODE_COLORS.keys())

# ---------------------------------------------------------------------------
# Node normalization helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    t = _STRIP_PREFIX.sub("", text.strip())
    t = _POSSESSIVE.sub("", t)          # "Service Provider's" → "Service Provider"
    return _MULTI_SPACE.sub(" ", t).lower()


def _build_surface_map(
    raw_mentions: list[tuple[str, str, int, str]],
) -> dict[str, str]:
    """
    Returns surface_to_canonical: maps any surface form (or its normalized
    version) to the canonical node label.

    The canonical label is the longest surface form seen for a given entity.
    Co-reference merging: if one normalized form is fully contained in another
    (within 40% length ratio) and both belong to the same NER type, the shorter
    is treated as an alias of the longer.
    """
    count_by_norm: dict[str, int] = defaultdict(int)
    surfaces_by_norm: dict[str, list[str]] = defaultdict(list)
    type_by_norm: dict[str, str] = {}

    for surface, ner_label, *_ in raw_mentions:
        n = _normalize(surface)
        count_by_norm[n] += 1
        surfaces_by_norm[n].append(surface.strip())
        type_by_norm.setdefault(n, ner_label)

    # Canonical label = longest surface form per normalized key
    canonical_by_norm: dict[str, str] = {
        n: max(ss, key=len) for n, ss in surfaces_by_norm.items()
    }

    # Cross-group merging: shorter norm aliases a longer norm of the same type
    norms_desc = sorted(canonical_by_norm, key=len, reverse=True)
    merged: dict[str, str] = {n: n for n in norms_desc}  # norm → parent norm

    for i, longer in enumerate(norms_desc):
        for shorter in norms_desc[i + 1 :]:
            if type_by_norm.get(longer) != type_by_norm.get(shorter):
                continue
            if shorter in longer and len(shorter) >= len(longer) * 0.4:
                if merged[shorter] == shorter:  # not already merged
                    merged[shorter] = longer

    # Resolve canonical label for each norm (follow merge chain)
    def _resolve(n: str) -> str:
        seen = set()
        while merged[n] != n:
            if n in seen:
                break
            seen.add(n)
            n = merged[n]
        return canonical_by_norm[n]

    surface_to_canonical: dict[str, str] = {}
    for n, surfs in surfaces_by_norm.items():
        canonical = _resolve(n)
        surface_to_canonical[n] = canonical
        for s in surfs:
            surface_to_canonical[s.strip()] = canonical

    return surface_to_canonical


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def _extract_raw_mentions(
    paragraphs: list[str], nlp
) -> list[tuple[str, str, int, str]]:
    """Returns [(surface_form, ner_label, para_idx, sentence_text), ...]."""
    mentions: list[tuple[str, str, int, str]] = []
    for para_idx, para in enumerate(paragraphs):
        doc = nlp(para)
        sent_spans = list(doc.sents)
        for ent in doc.ents:
            if ent.label_ not in _ALL_NER:
                continue
            sent_text = next(
                (s.text for s in sent_spans
                 if s.start_char <= ent.start_char < s.end_char),
                para[:200],
            )
            mentions.append((ent.text.strip(), ent.label_, para_idx, sent_text))
    return mentions


def _build_nodes(
    raw_mentions: list[tuple[str, str, int, str]],
    surface_map: dict[str, str],
    min_party_mentions: int = 2,
) -> dict[str, dict]:
    """Build node attribute dicts from raw NER mentions."""
    nodes: dict[str, dict] = {}

    for surface, ner_label, para_idx, sentence in raw_mentions:
        canonical = surface_map.get(_normalize(surface)) or surface_map.get(surface.strip())
        if canonical is None:
            continue
        node_type = _NER_TO_NODE_TYPE.get(ner_label, ner_label)

        if canonical not in nodes:
            nodes[canonical] = {
                "label": canonical,
                "type": node_type,
                "aliases": [],
                "para_indices": [],
                "first_context": sentence,
                "mention_count": 0,
            }

        d = nodes[canonical]
        d["mention_count"] += 1
        if para_idx not in d["para_indices"]:
            d["para_indices"].append(para_idx)
        if surface.strip() not in d["aliases"]:
            d["aliases"].append(surface.strip())

    # Filter nodes: PARTY needs min_party_mentions; other types need at least 2
    return {
        label: attrs
        for label, attrs in nodes.items()
        if (attrs["type"] == "PARTY" and attrs["mention_count"] >= min_party_mentions)
        or (attrs["type"] != "PARTY" and attrs["mention_count"] >= 2)
    }


def _extract_defined_terms(paragraphs: list[str]) -> list[tuple[str, int, str]]:
    results: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for idx, para in enumerate(paragraphs):
        for pat in _DEFINITION_PATTERNS:
            for m in pat.finditer(para):
                term = m.group("term").strip()
                if term.lower() not in seen:
                    seen.add(term.lower())
                    results.append((term, idx, para[:300]))
    return results


# ---------------------------------------------------------------------------
# Relationship extraction
# ---------------------------------------------------------------------------


def _nearest_ent_text(doc, root_i: int, ner_labels: set[str]) -> str | None:
    best: str | None = None
    best_dist = 9999
    for ent in doc.ents:
        if ent.label_ in ner_labels:
            dist = abs(ent.root.i - root_i)
            if dist < best_dist:
                best_dist = dist
                best = ent.text
    return best


def _dep_edges(
    paragraphs: list[str],
    known: set[str],
    surface_map: dict[str, str],
    nlp,
) -> list[dict]:
    edges: list[dict] = []

    for para_idx, para in enumerate(paragraphs):
        # Pre-split on semicolons to improve sentence boundary detection
        parts = re.split(r";\s+", para)
        for part in parts:
            doc = nlp(part)

            # Map span-root token index → canonical label
            tok_to_ent: dict[int, str] = {}
            for ent in doc.ents:
                canonical = (
                    surface_map.get(_normalize(ent.text))
                    or surface_map.get(ent.text.strip())
                )
                if canonical and canonical in known:
                    tok_to_ent[ent.root.i] = canonical

            if len(set(tok_to_ent.values())) < 2:
                continue

            for sent in doc.sents:
                s_tok = {i: v for i, v in tok_to_ent.items()
                         if sent.start <= i < sent.end}
                if len(set(s_tok.values())) < 2:
                    continue

                for tok in sent:
                    if tok.dep_ != "ROOT" or tok.pos_ not in ("VERB", "AUX"):
                        continue

                    subjects = _collect_ents_via_dep(tok, s_tok, {"nsubj", "nsubjpass"})
                    objects = _collect_ents_via_dep(tok, s_tok, {"dobj", "attr"})
                    # Prepositional objects
                    for child in tok.children:
                        if child.dep_ == "prep":
                            for pobj in child.children:
                                if pobj.dep_ == "pobj" and pobj.i in s_tok:
                                    objects.append(s_tok[pobj.i])
                                    _add_conjuncts(pobj, s_tok, objects)

                    if not subjects or not objects:
                        continue

                    relation = _VERB_MAP.get(tok.lemma_, "GENERIC")
                    if any(c.dep_ == "neg" for c in tok.children):
                        relation = _NEG_FLIP.get(relation, relation)

                    date_ref = _nearest_ent_text(doc, tok.i, {"DATE"})
                    amount_ref = _nearest_ent_text(doc, tok.i, {"MONEY"})
                    confidence = 0.8 if (subjects and objects and tok.dep_ == "ROOT") else 0.5

                    for subj in set(subjects):
                        for obj in set(objects):
                            if subj != obj:
                                edges.append(_make_edge(
                                    subj, obj, relation, tok.text,
                                    sent.text.strip(), para_idx,
                                    date_ref, amount_ref, "dependency", confidence,
                                ))

    return edges


def _collect_ents_via_dep(
    root_tok, tok_to_ent: dict[int, str], dep_labels: set[str]
) -> list[str]:
    result: list[str] = []
    for child in root_tok.children:
        if child.dep_ in dep_labels and child.i in tok_to_ent:
            result.append(tok_to_ent[child.i])
            _add_conjuncts(child, tok_to_ent, result)
    return result


def _add_conjuncts(tok, tok_to_ent: dict[int, str], result: list[str]) -> None:
    for child in tok.children:
        if child.dep_ == "conj" and child.i in tok_to_ent:
            result.append(tok_to_ent[child.i])


def _regex_edges(
    paragraphs: list[str],
    known: set[str],
    surface_map: dict[str, str],
) -> list[dict]:
    edges: list[dict] = []

    for para_idx, para in enumerate(paragraphs):
        for pat, relation, mode in _REGEX_PATTERNS:
            for m in pat.finditer(para):
                g = m.groupdict()
                ctx = para[max(0, m.start() - 40): m.end() + 40].strip()

                if mode == "ab":
                    a = _lookup(g.get("a", ""), surface_map)
                    b = _lookup(g.get("b", ""), surface_map)
                    if a and b and a in known and b in known and a != b:
                        edges.append(_make_edge(
                            a, b, relation, relation.lower().replace("_", " "),
                            ctx, para_idx, None, None, "regex", 0.75,
                        ))
                elif mode == "b_only":
                    b = _lookup(g.get("b", ""), surface_map)
                    if b and b in known:
                        # Find the first known PARTY in the same paragraph as source
                        a = _first_party_in_para(para, known, surface_map)
                        if a and a != b:
                            edges.append(_make_edge(
                                a, b, relation, "governed by",
                                ctx, para_idx, None, None, "regex", 0.7,
                            ))

    return edges


def _lookup(text: str, surface_map: dict[str, str]) -> str | None:
    t = text.strip()
    return surface_map.get(_normalize(t)) or surface_map.get(t)


def _first_party_in_para(
    para: str, known: set[str], surface_map: dict[str, str]
) -> str | None:
    """Return the first known label found verbatim in the paragraph text."""
    for label in known:
        if label in para:
            return label
    return None


def _make_edge(
    src: str, dst: str, relation: str, verb: str,
    sentence: str, para_idx: int,
    date_ref: str | None, amount_ref: str | None,
    method: str, confidence: float,
) -> dict:
    return {
        "src": src, "dst": dst,
        "relation": relation,
        "verb": verb,
        "sentence": sentence,
        "para_idx": para_idx,
        "date_ref": date_ref,
        "amount_ref": amount_ref,
        "extraction_method": method,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def build_graph(
    paragraphs: tuple[str, ...],
    model_name: str = "en_core_web_sm",
) -> tuple[dict, dict]:
    """
    Extract entities and relationships, returning a serialized graph dict and
    metadata dict.  Cached on (paragraphs, model_name).

    Serialized format (pickle-safe, cache-compatible):
        graph_dict = {
            "nodes": {label: attrs_dict, ...},
            "edges": [(src, dst, attrs_dict), ...]
        }
    """
    nlp = get_spacy_nlp(model_name)
    para_list = list(paragraphs)

    # Entities
    raw_mentions = _extract_raw_mentions(para_list, nlp)
    surface_map = _build_surface_map(raw_mentions)
    nodes = _build_nodes(raw_mentions, surface_map)

    # Defined terms
    for term, para_idx, ctx in _extract_defined_terms(para_list):
        if term not in nodes:
            nodes[term] = {
                "label": term,
                "type": "DEFINED_TERM",
                "aliases": [term],
                "para_indices": [para_idx],
                "first_context": ctx,
                "mention_count": 1,
            }
        surface_map.setdefault(_normalize(term), term)
        surface_map.setdefault(term, term)

    known = set(nodes)

    # Relationships
    all_edges = [
        e for e in (_dep_edges(para_list, known, surface_map, nlp)
                    + _regex_edges(para_list, known, surface_map))
        if e["src"] in known and e["dst"] in known
    ]

    meta = {
        "n_nodes": len(nodes),
        "n_edges": len(all_edges),
        "n_parties": sum(1 for v in nodes.values() if v["type"] == "PARTY"),
        "relation_types": sorted({e["relation"] for e in all_edges}),
    }
    graph_dict = {
        "nodes": nodes,
        "edges": [(e["src"], e["dst"], {k: v for k, v in e.items()
                                        if k not in ("src", "dst")})
                  for e in all_edges],
    }
    return graph_dict, meta


def deserialize_graph(graph_dict: dict) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for label, attrs in graph_dict["nodes"].items():
        G.add_node(label, **attrs)
    for src, dst, attrs in graph_dict["edges"]:
        G.add_edge(src, dst, **attrs)
    return G


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def build_figure(
    G: nx.MultiDiGraph,
    selected_types: set[str],
    selected_relations: set[str],
    min_confidence: float,
    parties_only: bool,
    show_labels: bool,
) -> "go.Figure":
    import plotly.graph_objects as go

    # Filter nodes
    if parties_only:
        keep_nodes = {n for n, d in G.nodes(data=True) if d.get("type") == "PARTY"}
    else:
        keep_nodes = {n for n, d in G.nodes(data=True) if d.get("type") in selected_types}

    # Filter edges
    keep_edges = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if d.get("relation", "GENERIC") in selected_relations
        and d.get("confidence", 0.0) >= min_confidence
        and u in keep_nodes and v in keep_nodes
    ]

    # Active nodes = nodes that appear in at least one edge + party nodes.
    # Isolated non-party nodes (DATE, MONEY, GPE with no edges) are excluded
    # so they don't dominate the layout as an unconnected outer ring.
    edge_nodes = {u for u, *_ in keep_edges} | {v for _, v, _ in keep_edges}
    party_nodes_set = {n for n, d in G.nodes(data=True) if d.get("type") == "PARTY"}
    active = edge_nodes | (keep_nodes & party_nodes_set)

    if not active:
        fig = go.Figure()
        fig.add_annotation(
            text="No nodes match the current filters.",
            showarrow=False, font=dict(size=14),
            xref="paper", yref="paper", x=0.5, y=0.5,
        )
        fig.update_layout(height=420, plot_bgcolor="white",
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    # Subgraph for layout
    Gf = nx.MultiDiGraph()
    for n in active:
        if n in G.nodes:
            Gf.add_node(n, **G.nodes[n])
    for u, v, d in keep_edges:
        Gf.add_edge(u, v, **d)

    # Layout — spring_layout with k scaled to node count spreads nodes well
    # and avoids the near-perfect circle that kamada_kawai produces on small
    # fully-connected graphs.
    n = Gf.number_of_nodes()
    k_val = max(1.5, 3.0 / (n ** 0.5))
    try:
        pos = nx.spring_layout(Gf, seed=7, k=k_val, iterations=120)
    except Exception:
        pos = nx.spring_layout(Gf, seed=42)

    node_order = list(Gf.nodes())
    traces: list = []
    annotations: list[dict] = []

    # ── Edge arrows + legend stubs ──
    edges_by_rel: dict[str, list[tuple]] = defaultdict(list)
    for u, v, d in keep_edges:
        edges_by_rel[d.get("relation", "GENERIC")].append((u, v, d))

    for rel, rel_edges in edges_by_rel.items():
        color = RELATION_COLORS.get(rel, "#aaaaaa")

        # Invisible legend stub
        traces.append(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=color, width=2.5),
            name=rel,
            legendgroup=rel,
            showlegend=True,
        ))

        for u, v, data in rel_edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]

            # Arrow annotation = the edge line + arrowhead.
            # standoff keeps the tip outside the node marker (20px diameter).
            annotations.append(dict(
                x=x1, y=y1,
                ax=x0, ay=y0,
                xref="x", yref="y",
                axref="x", ayref="y",
                text="",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=color,
                standoff=14,
            ))

            # Optional edge label at midpoint
            if show_labels:
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                label_text = data.get("verb", rel)
                annotations.append(dict(
                    x=mx, y=my,
                    xref="x", yref="y",
                    text=f"<i>{label_text}</i>",
                    showarrow=False,
                    font=dict(size=9, color=color),
                    bgcolor="rgba(255,255,255,0.75)",
                    borderpad=2,
                ))

    # ── Node trace ──
    nx_x = [pos[n][0] for n in node_order]
    nx_y = [pos[n][1] for n in node_order]
    node_colors = [NODE_COLORS.get(Gf.nodes[n].get("type", ""), "#999999") for n in node_order]

    hover_texts = []
    for n in node_order:
        d = Gf.nodes[n]
        aliases = ", ".join(d.get("aliases", [n])[:6])
        ctx = (d.get("first_context") or "")[:120]
        hover_texts.append(
            f"<b>{n}</b><br>"
            f"Type: {d.get('type', '?')}<br>"
            f"Mentions: {d.get('mention_count', '?')}<br>"
            f"Aliases: {aliases}<br>"
            f"<i>{ctx}</i>"
        )

    display_labels = [n if len(n) <= 22 else n[:20] + "…" for n in node_order]

    traces.append(go.Scatter(
        x=nx_x, y=nx_y,
        mode="markers+text",
        marker=dict(size=20, color=node_colors, line=dict(width=2, color="white")),
        text=display_labels,
        textposition="top center",
        textfont=dict(size=10),
        hovertext=hover_texts,
        hoverinfo="text",
        customdata=node_order,
        name="Nodes",
        showlegend=False,
    ))

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            showlegend=True,
            legend=dict(title=dict(text="Relationship"), itemclick="toggleothers"),
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=580,
            plot_bgcolor="white",
            paper_bgcolor="white",
            annotations=annotations,
        ),
    )
    return fig


def edges_to_dataframe(G: nx.MultiDiGraph, min_confidence: float = 0.0) -> "pd.DataFrame":
    import pandas as pd

    rows = []
    for u, v, data in G.edges(data=True):
        if data.get("confidence", 0.0) >= min_confidence:
            rows.append({
                "From": u,
                "Relation": data.get("relation", "GENERIC"),
                "To": v,
                "Verb": data.get("verb", ""),
                "¶": data.get("para_idx", ""),
                "Date": data.get("date_ref") or "",
                "Amount": data.get("amount_ref") or "",
                "Conf.": round(data.get("confidence", 0.0), 2),
                "Method": data.get("extraction_method", ""),
                "Sentence": (data.get("sentence") or "")[:150],
            })
    return pd.DataFrame(rows)
