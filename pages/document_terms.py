import io
import re

import pandas as pd
import streamlit as st

from src.comments.extract import extract_paragraphs
from src.utils.models import get_spacy_nlp
from src.utils.page import require_document


@st.cache_data(show_spinner=False)
def _extract_definitions(paragraphs: tuple[str, ...]) -> pd.DataFrame:
    patterns = [
        # "Term" means / shall mean / has the meaning
        re.compile(
            r'["\u201c\u2018](?P<term>[^"\u201d\u2019]{1,80})["\u201d\u2019]\s+(?:shall\s+)?(?:mean|have\s+the\s+meaning)',
            re.IGNORECASE,
        ),
        # defined as / referred to as / referred to herein as
        re.compile(
            r"\b(?P<term>[A-Z][A-Za-z\s]{1,60})\b\s+(?:is\s+)?(?:defined\s+as|referred\s+to\s+(?:herein\s+)?as)",
            re.IGNORECASE,
        ),
        # (hereinafter "Term") or (the "Term")
        re.compile(
            r'\((?:hereinafter\s+)?(?:the\s+)?["\u201c\u2018](?P<term>[^"\u201d\u2019]{1,80})["\u201d\u2019]\)',
            re.IGNORECASE,
        ),
    ]
    rows = []
    seen = set()
    for idx, para in enumerate(paragraphs):
        for pattern in patterns:
            for m in pattern.finditer(para):
                term = m.group("term").strip()
                if term.lower() in seen:
                    continue
                seen.add(term.lower())
                rows.append({"¶": idx, "Term": term, "Context": para})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _extract_entities(
    paragraphs: tuple[str, ...],
    labels: tuple[str, ...],
) -> pd.DataFrame:
    nlp = get_spacy_nlp()
    rows = []
    for idx, para in enumerate(paragraphs):
        doc = nlp(para)
        for ent in doc.ents:
            if ent.label_ in labels:
                rows.append(
                    {"¶": idx, "Value": ent.text, "Type": ent.label_, "Context": para}
                )
    return pd.DataFrame(rows)


_FALSE_DATE_PATTERNS = re.compile(
    r"^("
    r"[Ss]ection\s+"  # "Section 4.3.1"
    r"|\d{1,4}\.\d+"  # "4.3.1", "23.1"
    r"|\d{1,3}-\d{3,}"  # "23-1202", "15-123456"
    r"|\d{1,3}-\d+-[a-zA-Z]"  # "23-1202(c)"
    r"|§\s*\d+"  # "§ 23"
    r"|\d+\([a-zA-Z]\)"  # "4(b)"
    r"|\d+\)\s*(years?|months?|days?|weeks?|hours?)"  # "30) days", "2) years" (spaCy picks up the tail of parenthetical numbers)
    r"|\d+\s+(years?|months?|days?|weeks?|hours?)$"   # "30 days", "12 months" — plain durations, not calendar dates
    r")"
)

_NUMERIC_PATTERN = re.compile(
    r"(\$|€|£|USD|EUR|GBP|%|percent"
    r"|\b\d[\d,]*\.?\d*\s*(dollars?|cents?|million|billion|thousand|hundred)"
    r"|\b\d+\s*(days?|months?|years?|weeks?|hours?|minutes?|times?|attempts?|copies?|users?|seats?|units?)"
    r"|\bat\s+least\s+\d|\bup\s+to\s+\d|\bno\s+more\s+than\s+\d|\bno\s+less\s+than\s+\d"
    r")",
    re.IGNORECASE,
)
_SECTION_REF = re.compile(
    r"^\d{1,4}\.\d|\d{1,3}-\d{3,}|^§|^\d+\([a-zA-Z]\)|^[Ss]ection\s",
)
_OBLIGATION = re.compile(
    r"\b(shall|must|no later than|prior to|by|before|within|deadline|due|effective)\b",
    re.IGNORECASE,
)


def _clean_amounts(amounts_df: pd.DataFrame) -> pd.DataFrame:
    mask = ~amounts_df["Value"].apply(lambda v: bool(_SECTION_REF.search(v)))
    return amounts_df[mask].reset_index(drop=True)


def _clean_dates(dates_df: pd.DataFrame) -> pd.DataFrame:
    mask = ~dates_df["Value"].str.strip().apply(
        lambda v: bool(_FALSE_DATE_PATTERNS.match(v))
    )
    return dates_df[mask].reset_index(drop=True)


def _key_dates(dates_df: pd.DataFrame) -> pd.DataFrame:
    mask = dates_df["Context"].apply(lambda c: bool(_OBLIGATION.search(c)))
    return dates_df[mask].reset_index(drop=True)


def _render_expanded(df: pd.DataFrame, heading_col: str) -> None:
    lines: list[str] = []
    for _, row in df.iterrows():
        term = row[heading_col]
        context = re.sub(
            r"\b" + re.escape(term) + r"\b",
            lambda m: f"<mark>{m.group(0)}</mark>",
            row["Context"],
            flags=re.IGNORECASE,
        )
        lines.append(f"#### ¶{row['¶']} — {term}\n\n{context}")
    st.markdown("\n\n---\n\n".join(lines), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
file_bytes = require_document()

doc_paragraphs = extract_paragraphs(io.BytesIO(file_bytes))
paragraphs = tuple(p.strip() for p in doc_paragraphs.paragraphs if p and p.strip())

if not paragraphs:
    st.warning("No usable text found in the document.")
    st.stop()

st.subheader("Document Terms")
st.markdown(
    "Extracts key terms from the document: defined terms, dates, parties and entities, and numbers & amounts."
)

tab_defs, tab_dates, tab_parties, tab_amounts = st.tabs(
    ["Definitions", "Dates", "Parties & Entities", "Numbers & Amounts"]
)

with tab_defs:
    with st.spinner("Extracting definitions…"):
        defs_df = _extract_definitions(paragraphs)
    if defs_df.empty:
        st.info("No definitions found.")
    else:
        st.caption(f"{len(defs_df)} defined terms")
        st.dataframe(
            defs_df,
            width="stretch",
            hide_index=True,
            column_config={
                "¶": st.column_config.NumberColumn("¶", width="small"),
                "Term": st.column_config.TextColumn("Term", width="medium"),
                "Context": st.column_config.TextColumn("Context", width="large"),
            },
        )
        if st.checkbox("Show expanded view", key="dt_defs_expanded"):
            _render_expanded(defs_df, "Term")

with tab_dates:
    with st.spinner("Extracting dates…"):
        dates_df = _clean_dates(
            _extract_entities(paragraphs, ("DATE",)).drop(
                columns="Type", errors="ignore"
            )
        )
    if dates_df.empty:
        st.info("No dates found.")
    else:
        show_key_only = st.toggle(
            "Key dates only",
            value=False,
            help="Show only dates with obligation language (shall, must, by, no later than…)",
        )
        display_df = _key_dates(dates_df) if show_key_only else dates_df
        st.caption(f"{len(display_df)} dates")
        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            column_config={
                "¶": st.column_config.NumberColumn("¶", width="small"),
                "Value": st.column_config.TextColumn("Date", width="medium"),
                "Context": st.column_config.TextColumn("Context", width="large"),
            },
        )
        if st.checkbox("Show expanded view", key="dt_dates_expanded"):
            _render_expanded(display_df, "Value")

with tab_parties:
    with st.spinner("Extracting parties and entities…"):
        parties_df = _extract_entities(
            paragraphs, ("LAW", "PERSON", "ORG", "GPE", "LOC", "PRODUCT")
        )
    if parties_df.empty:
        st.info("No parties or entities found.")
    else:
        _TYPE_ORDER = ["LAW", "PERSON", "ORG", "GPE", "LOC", "PRODUCT"]
        available = [t for t in _TYPE_ORDER if t in parties_df["Type"].unique()]
        type_filter = st.pills(
            "Type",
            options=available,
            default=None,
            key="dt_party_filter",
            label_visibility="collapsed",
            selection_mode="single",
        )
        display_df = (
            parties_df[parties_df["Type"] == type_filter] if type_filter else parties_df
        )
        dedup = st.toggle("Unique only", value=False, key="dt_parties_dedup")
        if dedup:
            table_df = display_df.drop_duplicates(subset="Value")[
                ["Value", "Type"]
            ].reset_index(drop=True)
            st.caption(f"{len(table_df)} unique entities")
            st.dataframe(
                table_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Value": st.column_config.TextColumn("Entity", width="medium"),
                    "Type": st.column_config.TextColumn("Type", width="small"),
                },
            )
        else:
            st.caption(f"{len(display_df)} entities")
            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "¶": st.column_config.NumberColumn("¶", width="small"),
                    "Value": st.column_config.TextColumn("Entity", width="medium"),
                    "Type": st.column_config.TextColumn("Type", width="small"),
                    "Context": st.column_config.TextColumn("Context", width="large"),
                },
            )
            if st.checkbox("Show expanded view", key="dt_parties_expanded"):
                _render_expanded(display_df, "Value")

with tab_amounts:
    with st.spinner("Extracting amounts…"):
        amounts_df = _clean_amounts(
            _extract_entities(
                paragraphs, ("MONEY", "PERCENT", "QUANTITY", "CARDINAL")
            ).drop(columns="Type", errors="ignore")
        )
    if amounts_df.empty:
        st.info("No monetary amounts found.")
    else:
        st.caption(f"{len(amounts_df)} numbers & amounts")
        st.dataframe(
            amounts_df,
            width="stretch",
            hide_index=True,
            column_config={
                "¶": st.column_config.NumberColumn("¶", width="small"),
                "Value": st.column_config.TextColumn("Amount", width="medium"),
                "Context": st.column_config.TextColumn("Context", width="large"),
            },
        )
        if st.checkbox("Show expanded view", key="dt_amounts_expanded"):
            _render_expanded(amounts_df, "Value")
