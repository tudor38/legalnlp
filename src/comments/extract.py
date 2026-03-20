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

import io
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from pathlib import Path
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
    parent_id: Optional[str] = None
    replies: list["Comment"] = field(default_factory=list)
    context: Optional[CommentContext] = None

    def to_row(self) -> dict:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in ("replies", "context", "parent_id")
        } | {
            "parent_id": self.parent_id,
            "replies": len(self.replies),
            "selected": self.context.selected_text if self.context else None,
            "paragraph": self.context.paragraph_text if self.context else None,
            "sentences": self.context.sentences if self.context else [],
        }


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


def _is_libreoffice(zip_names: list[str], names_bytes: dict[str, bytes]) -> bool:
    """
    LibreOffice doesn't write commentsIds.xml and uses little-endian hex
    paraIds in commentsExtended.xml instead of real paragraph identifiers.
    We detect it via the app.xml producer string.
    """
    if "docProps/app.xml" not in zip_names:
        return False
    root = ET.fromstring(names_bytes["docProps/app.xml"])
    # app.xml has no namespace on <Application> in LibreOffice
    for elem in root.iter():
        if elem.tag.endswith("Application") and elem.text:
            return "libreoffice" in elem.text.lower()
    return False


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

        if cid is None:
            continue

        comments[cid] = Comment(id=cid, author=author, date=date, text=text)

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
    """
    root = ET.fromstring(xml_bytes)

    para_to_owner: dict[str, str] = {}
    for ci in root.findall(_tag(W14, "commentId")):
        para_id = ci.get(_tag(W14, "paraId"))
        owner_id = ci.get(_tag(W14, "paraIdOwner"))
        if para_id and owner_id:
            para_to_owner[para_id] = owner_id

    return para_to_owner


def _build_para_to_comment_from_document(xml_bytes: bytes) -> dict[str, str]:
    """
    Build {paraId: comment_id} by scanning document.xml for paragraphs
    that contain a <w:commentReference> element.
    Used for EXTENDED documents where comments.xml paragraphs lack w14:paraId.
    """
    root = ET.fromstring(xml_bytes)
    para_to_comment: dict[str, str] = {}

    for para in root.iter(_tag(W, "p")):
        para_id = para.get(_tag(W14, "paraId"))
        if para_id is None:
            continue
        ref = para.find(".//" + _tag(W, "commentReference"))
        if ref is not None:
            cid = ref.get(_tag(W, "id"))
            if cid:
                para_to_comment[para_id] = cid

    return para_to_comment


def _apply_extended(
    comments: dict[str, Comment],
    para_to_comment: dict[str, str],
    xml_bytes: bytes,
) -> None:
    root = ET.fromstring(xml_bytes)

    for ce in root.findall(_tag(W15, "commentEx")):
        para_id = ce.get(_tag(W15, "paraId"))
        parent_id = ce.get(_tag(W15, "paraIdParent"))
        done = ce.get(_tag(W15, "done"), "0") == "1"

        if para_id is None:
            continue

        cid = para_to_comment.get(para_id)

        # LibreOffice fallback: paraId is the comment w:id encoded as
        # a little-endian 32-bit hex string e.g. "01000000" → id "1"
        if cid is None:
            try:
                cid = str(int.from_bytes(bytes.fromhex(para_id), "little"))
            except (ValueError, TypeError):
                continue
            if cid not in comments:
                continue

        comments[cid].resolved = done

        if parent_id:
            parent_cid = para_to_comment.get(parent_id)
            if parent_cid is None:
                try:
                    parent_cid = str(int.from_bytes(bytes.fromhex(parent_id), "little"))
                except (ValueError, TypeError):
                    parent_cid = None
            if parent_cid and parent_cid in comments:
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

    We walk every <w:p> in document order and traverse descendants in DFS
    pre-order, tracking char_pos as we encounter <w:t> elements so that
    commentRangeStart/End markers give us exact character offsets.
    """
    root = ET.fromstring(xml_bytes)

    para_elements: list[ET.Element] = list(root.iter(_tag(W, "p")))

    open_ranges: dict[str, dict] = {}
    completed: dict[str, dict] = {}
    para_texts: list[str] = []

    for para_idx, para in enumerate(para_elements):
        char_pos = 0
        para_text_parts: list[str] = []

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
                        "end_char": char_pos,
                    }

            elif tag == _tag(W, "t"):
                text = elem.text or ""
                char_pos += len(text)
                para_text_parts.append(text)
                for acc in open_ranges.values():
                    acc["sel_chunks"].append(text)

        para_texts.append("".join(para_text_parts))

    contexts: dict[str, CommentContext] = {}

    for cid, info in completed.items():
        sp = info["start_para"]
        ep = info["end_para"]

        if sp == ep:
            para_text = para_texts[sp]
            sel_start = info["start_char"]
            sel_end = info["end_char"]
        else:
            para_text = "\n".join(para_texts[sp : ep + 1])
            sel_start = info["start_char"]
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
# Debug helper — writes key XML structures to /tmp/extract_comments_debug.log
# ---------------------------------------------------------------------------
def _debug_dump(label: str, content: str, mode: str = "a") -> None:
    with open("/tmp/extract_comments_debug.log", mode) as f:
        f.write(f"\n{'=' * 60}\n{label}\n{'=' * 60}\n")
        f.write(content)
        f.write("\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
type DocxSource = str | Path | io.IOBase


def extract_comments(
    docx: DocxSource, debug: bool = False
) -> tuple[list[Comment], WordVersion]:
    """
    Extract all comments from a .docx file.

    Parameters
    ----------
    docx  : path string, Path, or file-like object (e.g. st.UploadedFile)
    debug : if True, writes raw XML structures to
            /tmp/extract_comments_debug.log for troubleshooting

    Returns a list of top-level Comment objects (replies nested inside
    Comment.replies) and the detected WordVersion.
    """
    with zipfile.ZipFile(docx) as z:
        names = z.namelist()
        version = detect_version(names)

        # Read all relevant files up front — ZipFile members can only be
        # read once per open(), so we materialise bytes here and reuse them.
        comments_bytes = (
            z.read("word/comments.xml") if "word/comments.xml" in names else b""
        )
        extended_bytes = (
            z.read("word/commentsExtended.xml")
            if "word/commentsExtended.xml" in names
            else b""
        )
        ids_bytes = (
            z.read("word/commentsIds.xml") if "word/commentsIds.xml" in names else b""
        )
        document_bytes = (
            z.read("word/document.xml") if "word/document.xml" in names else b""
        )

    if debug:
        _debug_dump("version", version.name, mode="w")
        if comments_bytes:
            _debug_dump(
                "comments.xml (first 3000 chars)", comments_bytes.decode("utf-8")[:3000]
            )
        if extended_bytes:
            _debug_dump(
                "commentsExtended.xml",
                ET.tostring(ET.fromstring(extended_bytes), encoding="unicode"),
            )

    if not comments_bytes:
        return [], version

    comments, para_to_comment = _parse_comments(comments_bytes)

    if debug:
        _debug_dump("para_to_comment after _parse_comments", str(para_to_comment))

    if version == WordVersion.MODERN:
        para_to_owner = _parse_comments_ids(ids_bytes)
        for para_id, owner_para_id in para_to_owner.items():
            if para_id not in para_to_comment and owner_para_id in para_to_comment:
                para_to_comment[para_id] = para_to_comment[owner_para_id]
        _apply_extended(comments, para_to_comment, extended_bytes)

    elif version == WordVersion.EXTENDED:
        if document_bytes:
            para_to_comment.update(_build_para_to_comment_from_document(document_bytes))
        if debug:
            _debug_dump(
                "para_to_comment after _build_para_to_comment_from_document",
                str(para_to_comment),
            )
            ext_root = ET.fromstring(extended_bytes)
            _debug_dump(
                "commentsExtended paraId / paraIdParent pairs",
                str(
                    [
                        (ce.get(f"{{{W15}}}paraId"), ce.get(f"{{{W15}}}paraIdParent"))
                        for ce in ext_root.findall(_tag(W15, "commentEx"))
                    ]
                ),
            )
        _apply_extended(comments, para_to_comment, extended_bytes)

    # Document context is version-independent.
    if document_bytes:
        contexts = _parse_document_context(document_bytes)
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
    debug = "--debug" in sys.argv

    comments, version = extract_comments(path, debug=debug)

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


def extract_paragraphs(docx: DocxSource) -> list[str]:
    with zipfile.ZipFile(docx) as z:
        if "word/document.xml" not in z.namelist():
            return []
        root = ET.fromstring(z.read("word/document.xml"))
    return [
        "".join(t.text or "" for t in para.iter(_tag(W, "t")))
        for para in root.iter(_tag(W, "p"))
        if any(t.text for t in para.iter(_tag(W, "t")))  # skip empty paragraphs
    ]
