"""
Word document redline (tracked change) extractor.

Extracts insertions and deletions from word/document.xml.
Formatting-only changes (w:rPrChange, w:pPrChange) are not included
as they carry no text content.

Each Redline has:
    id          : w:id attribute from the change element
    author      : w:author
    date        : w:date (ISO 8601)
    kind        : "insertion" | "deletion"
    text        : the inserted or deleted text
    char_start  : start offset of the change within its paragraph (inclusive)
    char_end    : end offset of the change within its paragraph (exclusive)
    context     : RedlineContext — paragraph index, full paragraph text,
                  and SentenceSpan objects with their own paragraph-relative
                  offsets

All character offsets are paragraph-relative (reset to 0 at each <w:p>).

Version handling
----------------
Tracked changes live entirely in word/document.xml and have not changed
schema across Word versions, so no version-specific parsing is needed.
WordVersion is still detected and returned for consistency with the
comments API.
"""

import io
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields
from enum import Enum, auto
from pathlib import Path
from typing import Literal, Optional

import spacy

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


# ---------------------------------------------------------------------------
# Namespaces  (shared with extract_comments)
# ---------------------------------------------------------------------------
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W14 = "http://schemas.microsoft.com/office/word/2010/wordml"
W15 = "http://schemas.microsoft.com/office/word/2012/wordml"


def _tag(ns: str, local: str) -> str:
    return f"{{{ns}}}{local}"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
class WordVersion(Enum):
    LEGACY = auto()
    EXTENDED = auto()
    MODERN = auto()


@dataclass
class Span:
    """A [start, end) character range, paragraph-relative."""

    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and self.end > other.start


@dataclass
class SentenceSpan:
    """A sentence and its paragraph-relative character span."""

    text: str
    span: Span


@dataclass
class RedlineContext:
    """Surrounding text context for a tracked change."""

    para_idx: int  # 0-based paragraph index in document
    paragraph_text: str  # full text of the containing paragraph
    sentences: list[SentenceSpan]  # sentences overlapping the change


@dataclass
class Redline:
    id: str
    author: str
    date: str
    kind: Literal["insertion", "deletion"]
    text: str
    char_start: int  # paragraph-relative, inclusive
    char_end: int  # paragraph-relative, exclusive
    context: Optional[RedlineContext] = None

    @property
    def span(self) -> Span:
        """Convenience accessor for the change span."""
        return Span(self.char_start, self.char_end)

    def to_row(self) -> dict:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in ("context",)
        } | {
            "para_idx": self.context.para_idx if self.context else None,
            "paragraph": self.context.paragraph_text if self.context else None,
            "sentences": [s.text for s in self.context.sentences]
            if self.context
            else [],
        }


# ---------------------------------------------------------------------------
# Version detection  (mirrors extract_comments.detect_version)
# ---------------------------------------------------------------------------
def detect_version(zip_names: list[str]) -> WordVersion:
    has_extended = "word/commentsExtended.xml" in zip_names
    has_ids = "word/commentsIds.xml" in zip_names
    if has_extended and has_ids:
        return WordVersion.MODERN
    if has_extended:
        return WordVersion.EXTENDED
    return WordVersion.LEGACY


# ---------------------------------------------------------------------------
# Sentence helper — returns SentenceSpan instead of plain str
# ---------------------------------------------------------------------------
def _find_sentences_containing(
    text: str, sel_start: int, sel_end: int
) -> list[SentenceSpan]:
    """
    Return every sentence in text whose span overlaps [sel_start, sel_end).
    Each result carries the sentence text and its paragraph-relative Span.
    """
    if not text or sel_start >= sel_end:
        return []
    doc = nlp(text)
    return [
        SentenceSpan(
            text=sent.text.strip(),
            span=Span(sent.start_char, sent.end_char),
        )
        for sent in doc.sents
        if sent.start_char < sel_end and sent.end_char > sel_start
    ]


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------
def _parse_redlines(xml_bytes: bytes) -> list[Redline]:
    """
    Walk word/document.xml and extract all tracked insertions and deletions.

    Strategy
    --------
    We iterate every <w:p> in document order.  Within each paragraph we
    collect the full paragraph text (from all <w:t> and <w:delText> nodes,
    so that both inserted and deleted text contribute to the context) while
    tracking character offsets for each change range.

    For each <w:ins> or <w:del> element we record:
        - its attributes (id, author, date)
        - the text of its descendant <w:t> / <w:delText> nodes
        - the [start, end) character offsets within the paragraph text

    After collecting all paragraphs we build RedlineContext objects.

    Note on paragraph text composition
    -----------------------------------
    We include both <w:t> (normal / inserted runs) and <w:delText> (deleted
    runs) when building paragraph_text so that the surrounding sentence
    context reads naturally.  The change text itself is extracted only from
    the appropriate element type per change kind.

    Note on char_pos and <w:ins> / <w:del> traversal
    -------------------------------------------------
    DFS pre-order means we visit <w:ins> before its <w:t> children, and
    <w:del> before its <w:delText> children.  We record char_pos at the
    moment we encounter the change element (before its children are visited),
    which is the correct start offset.  The end offset is start + len(text)
    since we know the full text at that point.  We do NOT advance char_pos
    inside the change element handler — the <w:t> / <w:delText> handlers
    below advance it as they are visited, keeping char_pos consistent for
    any elements that follow.
    """
    root = ET.fromstring(xml_bytes)

    redlines: list[Redline] = []
    para_texts: list[str] = []

    # {redline_id: {"start": int, "end": int, "para_idx": int}}
    change_spans: dict[str, dict] = {}

    for para_idx, para in enumerate(root.iter(_tag(W, "p"))):
        char_pos = 0
        para_text_parts: list[str] = []

        for elem in para.iter():
            tag = elem.tag

            if tag in (_tag(W, "ins"), _tag(W, "del")):
                rid = elem.get(_tag(W, "id"))
                author = elem.get(_tag(W, "author"), "")
                date = elem.get(_tag(W, "date"), "")
                kind: Literal["insertion", "deletion"] = (
                    "insertion" if tag == _tag(W, "ins") else "deletion"
                )

                text_tag = _tag(W, "t") if kind == "insertion" else _tag(W, "delText")
                text = "".join(t.text or "" for t in elem.iter(text_tag))

                if rid is None:
                    continue

                change_spans[rid] = {
                    "start": char_pos,
                    "end": char_pos + len(text),
                    "para_idx": para_idx,
                }
                redlines.append(
                    Redline(
                        id=rid,
                        author=author,
                        date=date,
                        kind=kind,
                        text=text,
                        char_start=char_pos,
                        char_end=char_pos + len(text),
                    )
                )

            elif tag == _tag(W, "t"):
                t = elem.text or ""
                char_pos += len(t)
                para_text_parts.append(t)

            elif tag == _tag(W, "delText"):
                t = elem.text or ""
                char_pos += len(t)
                para_text_parts.append(t)

        para_texts.append("".join(para_text_parts))

    # ------------------------------------------------------------------
    # Attach RedlineContext to each Redline.
    # ------------------------------------------------------------------
    rid_to_redline = {r.id: r for r in redlines}

    for rid, span in change_spans.items():
        if rid not in rid_to_redline:
            continue

        para_text = para_texts[span["para_idx"]]
        sel_start = span["start"]
        sel_end = span["end"]

        rid_to_redline[rid].context = RedlineContext(
            para_idx=span["para_idx"],
            paragraph_text=para_text,
            sentences=_find_sentences_containing(para_text, sel_start, sel_end),
        )

    return redlines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
type DocxSource = str | Path | io.IOBase


def extract_redlines(docx: DocxSource) -> tuple[list[Redline], WordVersion]:
    """
    Extract all tracked changes from a .docx file.

    Returns a list of Redline objects and the detected WordVersion.

    Each Redline has:
        .id         : change id
        .author     : who made the change
        .date       : when (ISO 8601 string)
        .kind       : "insertion" or "deletion"
        .text       : the inserted or deleted text
        .char_start : start offset within containing paragraph (inclusive)
        .char_end   : end offset within containing paragraph (exclusive)
        .span       : Span(char_start, char_end) convenience property
        .context    : RedlineContext with para_idx, paragraph_text,
                      and sentences (list[SentenceSpan])

    All offsets are paragraph-relative.
    Formatting-only changes are not included.
    """
    with zipfile.ZipFile(docx) as z:
        names = z.namelist()
        version = detect_version(names)
        document_bytes = (
            z.read("word/document.xml") if "word/document.xml" in names else b""
        )

    if not document_bytes:
        return [], version

    return _parse_redlines(document_bytes), version


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "document.docx"
    redlines, version = extract_redlines(path)

    print(f"Format detected  : {version.name}")
    print(f"Redlines found   : {len(redlines)}\n")

    for r in redlines:
        print(f"[{r.kind.upper()}] ({r.id}) {r.author} @ {r.date}")
        print(f"  Text       : {r.text!r}")
        print(f"  Span       : [{r.char_start}, {r.char_end})")
        if r.context:
            print(f"  Para idx   : {r.context.para_idx}")
            print(f"  Paragraph  : {r.context.paragraph_text!r}")
            for s in r.context.sentences:
                print(f"  Sentence   : {s.text!r}  [{s.span.start}, {s.span.end})")
        print()
