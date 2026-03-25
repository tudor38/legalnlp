import hashlib
import io
import re

import pandas as pd
import spacy
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
        # (hereinafter "Term"), (the "Term"), (together, the "Parties"), (each, a "Party"), etc.
        re.compile(
            r'\([^"\u201c\u2018]{0,50}["\u201c\u2018](?P<term>[^"\u201d\u2019]{1,80})["\u201d\u2019]\)',
            re.IGNORECASE,
        ),
        # (together, the Parties), (hereinafter, the Disclosing Party) — no quotes, Title Case term
        re.compile(
            r'\([^")]{0,60}(?:the|an?)\s+(?P<term>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s*[,;.]?\s*\)',
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
                rows.append({"Para": idx, "Term": term, "Context": para})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _extract_entities(
    paragraphs: tuple[str, ...],
    labels: tuple[str, ...],
    model_name: str = "en_core_web_sm",
) -> pd.DataFrame:
    nlp = get_spacy_nlp(model_name)
    rows = []
    for idx, para in enumerate(paragraphs):
        doc = nlp(para)
        for ent in doc.ents:
            if ent.label_ in labels:
                rows.append(
                    {"Para": idx, "Value": ent.text, "Type": ent.label_, "Context": para}
                )
    return pd.DataFrame(rows)


_MONEY_RE = re.compile(
    r"(?:"
    r"(?:USD|EUR|GBP|CHF|CAD|AUD|JPY)\s+\d{1,3}(?:,\d{3})*(?:\.\d+)?"   # USD 8,500
    r"|[\$€£]\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?"                            # $8,500 / € 100,000
    r"|\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:USD|EUR|GBP|CHF)"              # 8,500 USD
    r"|\d+(?:\.\d+)?\s*(?:%|percent|per\s+cent)"                         # 18%, 1.5 %
    r")",
    re.IGNORECASE,
)


@st.cache_data(show_spinner=False)
def _extract_money(paragraphs: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for idx, para in enumerate(paragraphs):
        seen: set[str] = set()
        for m in _MONEY_RE.finditer(para):
            val = m.group(0).strip()
            if val not in seen:
                seen.add(val)
                rows.append({"Para": idx, "Value": val, "Context": para})
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
            r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])",
            lambda m: f"<mark>{m.group(0)}</mark>",
            row["Context"],
            flags=re.IGNORECASE,
        )
        lines.append(f"#### Para {row['Para']} — {term}\n\n{context}")
    st.markdown("\n\n---\n\n".join(lines), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _get_paragraphs(file_bytes: bytes) -> tuple[str, ...]:
    doc = extract_paragraphs(io.BytesIO(file_bytes))
    return tuple(p.strip() for p in doc.paragraphs if p and p.strip())


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
file_bytes = require_document()

paragraphs = _get_paragraphs(file_bytes)

if not paragraphs:
    st.warning("No usable text found in the document.")
    st.stop()

with st.sidebar:
    _all_models = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"]
    _installed = [m for m in _all_models if spacy.util.is_package(m)]
    _saved = st.session_state.get("_dt_spacy_model_pref")
    _index = _installed.index(_saved) if _saved in _installed else 0
    spacy_model = st.selectbox(
        "spaCy model",
        options=_installed,
        index=_index,
        help="Larger models improve entity recognition quality. md/lg are a good balance; trf is most accurate but slower.",
    )
    st.session_state["_dt_spacy_model_pref"] = spacy_model

st.subheader("Document Terms")
st.markdown(
    "Extracts key terms from the document: defined terms, dates, parties and entities, monetary values, and numbers."
)

# Recompute only when the document or model changes
_cache_key = hashlib.md5(file_bytes).hexdigest() + "|" + spacy_model

if st.session_state.get("_doc_terms_key") != _cache_key:
    try:
        with st.spinner("Extracting document terms…"):
            defs_df = _extract_definitions(paragraphs)
            dates_df = _clean_dates(
                _extract_entities(paragraphs, ("DATE",), spacy_model).drop(columns="Type", errors="ignore")
            )
            parties_df = _extract_entities(
                paragraphs, ("LAW", "PERSON", "ORG", "GPE", "LOC", "PRODUCT"), spacy_model
            )
            money_df = _extract_money(paragraphs)
            numbers_df = _clean_amounts(
                _extract_entities(paragraphs, ("QUANTITY", "CARDINAL"), spacy_model).drop(columns="Type", errors="ignore")
            )
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    st.session_state.update({
        "_doc_terms_key": _cache_key,
        "_doc_terms_defs": defs_df,
        "_doc_terms_dates": dates_df,
        "_doc_terms_parties": parties_df,
        "_doc_terms_money": money_df,
        "_doc_terms_numbers": numbers_df,
    })
else:
    defs_df = st.session_state["_doc_terms_defs"]
    dates_df = st.session_state["_doc_terms_dates"]
    parties_df = st.session_state["_doc_terms_parties"]
    money_df = st.session_state["_doc_terms_money"]
    numbers_df = st.session_state["_doc_terms_numbers"]

tab_defs, tab_dates, tab_parties, tab_money, tab_numbers = st.tabs(
    ["Definitions", "Dates", "Parties & Entities", "Money", "Numbers"]
)

with tab_defs:
    if defs_df.empty:
        st.info("No definitions found.")
    else:
        st.caption(f"{len(defs_df)} defined terms")
        st.dataframe(
            defs_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Para": st.column_config.NumberColumn("Para", width="small"),
                "Term": st.column_config.TextColumn("Term", width="medium"),
                "Context": st.column_config.TextColumn("Context", width="large"),
            },
        )
        if st.checkbox("Show expanded view", key="dt_defs_expanded"):
            _render_expanded(defs_df, "Term")

with tab_dates:
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
                "Para": st.column_config.NumberColumn("Para", width="small"),
                "Value": st.column_config.TextColumn("Date", width="medium"),
                "Context": st.column_config.TextColumn("Context", width="large"),
            },
        )
        if st.checkbox("Show expanded view", key="dt_dates_expanded"):
            _render_expanded(display_df, "Value")

with tab_parties:
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
                    "Para": st.column_config.NumberColumn("Para", width="small"),
                    "Value": st.column_config.TextColumn("Entity", width="medium"),
                    "Type": st.column_config.TextColumn("Type", width="small"),
                    "Context": st.column_config.TextColumn("Context", width="large"),
                },
            )
            if st.checkbox("Show expanded view", key="dt_parties_expanded"):
                _render_expanded(display_df, "Value")

with tab_money:
    if money_df.empty:
        st.info("No monetary values found.")
    else:
        st.caption(f"{len(money_df)} monetary values")
        st.dataframe(
            money_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Para": st.column_config.NumberColumn("Para", width="small"),
                "Value": st.column_config.TextColumn("Amount", width="medium"),
                "Context": st.column_config.TextColumn("Context", width="large"),
            },
        )
        if st.checkbox("Show expanded view", key="dt_money_expanded"):
            _render_expanded(money_df, "Value")

with tab_numbers:
    if numbers_df.empty:
        st.info("No numbers found.")
    else:
        st.caption(f"{len(numbers_df)} numbers")
        st.dataframe(
            numbers_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Para": st.column_config.NumberColumn("Para", width="small"),
                "Value": st.column_config.TextColumn("Value", width="medium"),
                "Context": st.column_config.TextColumn("Context", width="large"),
            },
        )
        if st.checkbox("Show expanded view", key="dt_numbers_expanded"):
            _render_expanded(numbers_df, "Value")
