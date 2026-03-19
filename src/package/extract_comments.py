"""
Word document comment extractor.

Supports three format generations, detected automatically:

  LEGACY   (Word 2007-2010)  word/comments.xml only
                             → author, date, text
  EXTENDED (Word 2013-2016)  + word/commentsExtended.xml
                             → + resolved status, reply threading
  MODERN   (Word 2016+/365)  + word/commentsIds.xml
                             → same, but paraId→commentId mapping is
                               read from commentsIds.xml instead of
                               being inferred from paragraph order

Document context (all versions):
  word/document.xml is always parsed to extract, per comment:
    - selected_text  : the exact text the comment is anchored to
    - paragraph_text : full text of the paragraph(s) containing the range
    - sentences      : sentence(s) in paragraph_text that overlap the range

  Note: comment ranges anchored in headers, footers, footnotes, or endnotes
  are not currently resolved (context will be None for those comments).
"""

import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import spacy


nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


# ---------------------------------------------------------------------------
# Namespaces
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
    LEGACY = auto()  # comments.xml only
    EXTENDED = auto()  # + commentsExtended.xml
    MODERN = auto()  # + commentsIds.xml


@dataclass
class CommentContext:
    """Text context of a comment's anchor point in the document body."""

    selected_text: str  # exact text between commentRangeStart/End
    paragraph_text: str  # full text of the containing paragraph(s)
    sentences: list[str]  # sentence(s) that overlap the selected range


@dataclass
class Comment:
    id: str
    author: str
    date: str
    text: str
    resolved: bool = False
    parent_id: Optional[str] = None  # comment id of parent, if a reply
    replies: list["Comment"] = field(default_factory=list)
    context: Optional[CommentContext] = None  # None for header/footer comments


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------
def detect_version(zip_names: list[str]) -> WordVersion:
    """
    Infer the Word format generation from which XML files are present.

    commentsIds.xml was introduced in Word 2016 alongside the modern
    threaded-comments UI; commentsExtended.xml appeared in Word 2013.
    """
    has_extended = "word/commentsExtended.xml" in zip_names
    has_ids = "word/commentsIds.xml" in zip_names

    if has_extended and has_ids:
        return WordVersion.MODERN
    if has_extended:
        return WordVersion.EXTENDED
    return WordVersion.LEGACY


# ---------------------------------------------------------------------------
# Parsers — one per file, version-agnostic internally
# ---------------------------------------------------------------------------
def _parse_comments(xml_bytes: bytes) -> tuple[dict[str, Comment], dict[str, str]]:
    """
    Parse word/comments.xml.

    Returns
    -------
    comments        : {comment_id: Comment}
    para_to_comment : {paraId: comment_id}
                      Built from the w14:paraId attribute on each comment's
                      first <w:p>.  Used by LEGACY and EXTENDED; MODERN
                      supplements this with commentsIds.xml.
    """
    root = ET.fromstring(xml_bytes)

    comments: dict[str, Comment] = {}
    para_to_comment: dict[str, str] = {}

    for c in root.findall(_tag(W, "comment")):
        cid = c.get(_tag(W, "id"))
        author = c.get(_tag(W, "author"), "")
        date = c.get(_tag(W, "date"), "")
        text = "".join(t.text or "" for t in c.iter(_tag(W, "t")))

        comments[cid] = Comment(id=cid, author=author, date=date, text=text)

        # Map the first paragraph's paraId to this comment.
        # Word stores the stable per-paragraph identity in w14:paraId.
        first_para = c.find(_tag(W, "p"))
        if first_para is not None:
            para_id = first_para.get(_tag(W14, "paraId"))
            if para_id:
                para_to_comment[para_id] = cid

    return comments, para_to_comment


def _parse_comments_ids(xml_bytes: bytes) -> dict[str, str]:
    """
    Parse word/commentsIds.xml  (MODERN only).

    Each <w14:commentId> element links a paragraph's paraId to the root
    paraId of its thread:

        w14:paraId       – paraId of one paragraph inside the comment body
        w14:paraIdOwner  – paraId of the thread-root comment's paragraph

    Returns a flat {paraId: paraId_of_root} map so that any paragraph in a
    thread can be resolved back to the root comment.  Callers then look up
    the root paraId in para_to_comment to get the actual comment id.
    """
    root = ET.fromstring(xml_bytes)

    para_to_owner: dict[str, str] = {}
    for ci in root.findall(_tag(W14, "commentId")):
        para_id = ci.get(_tag(W14, "paraId"))
        owner_id = ci.get(_tag(W14, "paraIdOwner"))
        if para_id and owner_id:
            para_to_owner[para_id] = owner_id

    return para_to_owner


def _apply_extended(
    comments: dict[str, Comment],
    para_to_comment: dict[str, str],
    xml_bytes: bytes,
) -> None:
    """
    Parse word/commentsExtended.xml and mutate comments in-place.

    Each <w15:commentEx> carries:
        w15:paraId        – identifies which comment this record belongs to
        w15:paraIdParent  – present on replies; paraId of the parent comment
        w15:done          – "1" if the thread is marked resolved

    Used by both EXTENDED and MODERN (identical schema in both).
    """
    root = ET.fromstring(xml_bytes)

    for ce in root.findall(_tag(W15, "commentEx")):
        para_id = ce.get(_tag(W15, "paraId"))
        parent_id = ce.get(_tag(W15, "paraIdParent"))
        done = ce.get(_tag(W15, "done"), "0") == "1"

        cid = para_to_comment.get(para_id)
        if cid is None:
            continue

        comments[cid].resolved = done

        if parent_id:
            parent_cid = para_to_comment.get(parent_id)
            if parent_cid:
                comments[cid].parent_id = parent_cid


# ---------------------------------------------------------------------------
# Document context — selected text, paragraph, and sentences
# ---------------------------------------------------------------------------
def _find_sentences_containing(text: str, sel_start: int, sel_end: int) -> list[str]:
    if not text or sel_start >= sel_end:
        return []
    doc = nlp(text)
    return [
        sent.text.strip()
        for sent in doc.sents
        if sent.start_char < sel_end and sent.end_char > sel_start
    ]


def _parse_document_context(xml_bytes: bytes) -> dict[str, CommentContext]:
    """
    Parse word/document.xml and extract a CommentContext for each comment id.

    Strategy
    --------
    We iterate every <w:p> paragraph element in document order.  Within each
    paragraph, we traverse all descendants in DFS pre-order — the same order
    they appear in the serialised XML — which gives us interleaved text and
    comment-range markers in the correct sequence:

        <w:r><w:t>Before </w:t></w:r>          char_pos advances to 7
        <w:commentRangeStart w:id="1"/>         range 1 opens  @ char_pos=7
        <w:r><w:t>Selected</w:t></w:r>          char_pos advances to 15
        <w:commentRangeEnd   w:id="1"/>         range 1 closes @ char_pos=15

    selected_text = paragraph_text[7:15] = "Selected".

    Multi-paragraph ranges
    ----------------------
    Word allows a comment range to span paragraph boundaries.  When an open
    range carries over to the next paragraph, we inject a "\n" separator into
    both the selected-text accumulator and the joined paragraph_text string so
    that character offsets remain consistent across both.

    Offsets recorded at range-open / range-close are relative to their
    respective paragraphs.  After collecting all paragraph texts, end-offsets
    are converted to positions within the joined paragraph_text string.
    """
    root = ET.fromstring(xml_bytes)

    # All paragraphs in the document, in document order.
    # root.iter() gives DFS pre-order, covering body, table-cell, text-box
    # paragraphs, etc.  <w:p> never contains descendant <w:p> elements in
    # standard OOXML, so para.iter() is safe for single-paragraph traversal.
    para_elements: list[ET.Element] = list(root.iter(_tag(W, "p")))

    # {cid: {"start_para": int, "start_char": int, "sel_chunks": list[str]}}
    open_ranges: dict[str, dict] = {}

    # {cid: {"selected":   str,
    #         "start_para": int, "start_char": int,
    #         "end_para":   int, "end_char":   int}}
    completed: dict[str, dict] = {}

    para_texts: list[str] = []

    for para_idx, para in enumerate(para_elements):
        char_pos = 0
        para_text_parts: list[str] = []

        # Ranges that are already open receive a paragraph separator so that
        # selected_text and paragraph_text are built from the same characters.
        for acc in open_ranges.values():
            acc["sel_chunks"].append("\n")

        for elem in para.iter():
            tag = elem.tag

            if tag == _tag(W, "commentRangeStart"):
                cid = elem.get(_tag(W, "id"))
                if cid:
                    open_ranges[cid] = {
                        "start_para": para_idx,
                        "start_char": char_pos,
                        "sel_chunks": [],
                    }

            elif tag == _tag(W, "commentRangeEnd"):
                cid = elem.get(_tag(W, "id"))
                if cid and cid in open_ranges:
                    acc = open_ranges.pop(cid)
                    completed[cid] = {
                        "selected": "".join(acc["sel_chunks"]),
                        "start_para": acc["start_para"],
                        "start_char": acc["start_char"],
                        "end_para": para_idx,
                        "end_char": char_pos,  # offset within this paragraph
                    }

            elif tag == _tag(W, "t"):
                text = elem.text or ""
                char_pos += len(text)
                para_text_parts.append(text)
                for acc in open_ranges.values():
                    acc["sel_chunks"].append(text)

        para_texts.append("".join(para_text_parts))

    # ------------------------------------------------------------------
    # Build CommentContext objects now that all para_texts are known.
    # ------------------------------------------------------------------
    contexts: dict[str, CommentContext] = {}

    for cid, info in completed.items():
        sp = info["start_para"]
        ep = info["end_para"]

        if sp == ep:
            # Single-paragraph range: offsets are directly within para_texts[sp].
            para_text = para_texts[sp]
            sel_start = info["start_char"]
            sel_end = info["end_char"]
        else:
            # Multi-paragraph: join with the same "\n" separators injected into
            # sel_chunks so that character offsets stay consistent.
            para_text = "\n".join(para_texts[sp : ep + 1])
            sel_start = info["start_char"]  # offset within para_texts[sp], which
            # starts at position 0 in para_text
            # Convert end_char (within para_texts[ep]) to an offset within para_text
            # by summing lengths of all preceding paragraphs plus one "\n" each.
            offset_to_ep = sum(len(para_texts[i]) + 1 for i in range(sp, ep))
            sel_end = offset_to_ep + info["end_char"]

        contexts[cid] = CommentContext(
            selected_text=info["selected"],
            paragraph_text=para_text,
            sentences=_find_sentences_containing(para_text, sel_start, sel_end),
        )

    return contexts


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------
def _build_tree(comments: dict[str, Comment]) -> list[Comment]:
    """Nest replies under their parents; return only top-level comments."""
    top_level: list[Comment] = []

    for c in comments.values():
        if c.parent_id and c.parent_id in comments:
            comments[c.parent_id].replies.append(c)
        else:
            top_level.append(c)

    return top_level


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_comments(docx_path: str) -> tuple[list[Comment], WordVersion]:
    """
    Extract all comments from a .docx file.

    Returns a list of top-level Comment objects (replies nested inside
    Comment.replies) and the detected WordVersion.

    Every comment has a .context attribute (CommentContext) populated with
    the selected text, containing paragraph(s), and overlapping sentence(s).
    context is None only for comments anchored in headers, footers, or other
    XML parts not covered by word/document.xml.

    Version-specific behaviour
    --------------------------
    LEGACY   No resolved status or reply threading.  Returns flat list;
             resolved=False and parent_id=None for all comments.

    EXTENDED paraId→commentId mapping inferred from comments.xml, then
             extended metadata applied from commentsExtended.xml.

    MODERN   commentsIds.xml read for the authoritative paraId→ownerParaId
             map, used to supplement the inferred mapping before applying
             commentsExtended.xml.
    """
    with zipfile.ZipFile(docx_path) as z:
        names = z.namelist()
        version = detect_version(names)

        if "word/comments.xml" not in names:
            return [], version

        comments, para_to_comment = _parse_comments(z.read("word/comments.xml"))

        if version == WordVersion.MODERN:
            para_to_owner = _parse_comments_ids(z.read("word/commentsIds.xml"))
            for para_id, owner_para_id in para_to_owner.items():
                if para_id not in para_to_comment and owner_para_id in para_to_comment:
                    para_to_comment[para_id] = para_to_comment[owner_para_id]
            _apply_extended(
                comments, para_to_comment, z.read("word/commentsExtended.xml")
            )

        elif version == WordVersion.EXTENDED:
            _apply_extended(
                comments, para_to_comment, z.read("word/commentsExtended.xml")
            )

        # LEGACY: nothing more to do for threading/resolved.

        # Document context is version-independent.
        if "word/document.xml" in names:
            contexts = _parse_document_context(z.read("word/document.xml"))
            for cid, ctx in contexts.items():
                if cid in comments:
                    comments[cid].context = ctx

    return _build_tree(comments), version


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "document.docx"
    comments, version = extract_comments(path)

    print(f"Format detected : {version.name}")
    print(f"Comments found  : {len(comments)}\n")

    for comment in comments:
        status = "RESOLVED" if comment.resolved else "OPEN"
        print(f"[{status}] ({comment.id}) {comment.author} @ {comment.date}")
        print(f"  Comment  : {comment.text}")
        if comment.context:
            print(f"  Selected : {comment.context.selected_text!r}")
            print(f"  Paragraph: {comment.context.paragraph_text!r}")
            for sent in comment.context.sentences:
                print(f"  Sentence : {sent!r}")
        for reply in comment.replies:
            r_status = "RESOLVED" if reply.resolved else "OPEN"
            print(f"  ↳ [{r_status}] ({reply.id}) {reply.author} @ {reply.date}")
            print(f"      Comment  : {reply.text}")
            if reply.context:
                print(f"      Selected: {reply.context.selected_text!r}")
        print()
