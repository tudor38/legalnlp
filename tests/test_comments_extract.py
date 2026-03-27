"""
Tests for the comment extractor in src/comments/extract.py.

Tests operate directly on the private XML-parsing functions using minimal
inline fixtures, plus the public extract_paragraphs function which requires
a real in-memory .docx zip.
"""

import io
import zipfile

from src.comments.extract import (
    Comment,
    DocumentParagraphs,
    _build_tree,
    _parse_comments,
    _parse_document_context,
    extract_paragraphs,
)

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W14 = "http://schemas.microsoft.com/office/word/2010/wordml"


# ---------------------------------------------------------------------------
# XML fixture helpers
# ---------------------------------------------------------------------------


def _comments_xml(body: str) -> bytes:
    return (f'<w:comments xmlns:w="{W}" xmlns:w14="{W14}">{body}</w:comments>').encode()


def _comment_el(cid: str, author: str, text: str, para_id: str = "") -> str:
    para_id_attr = f' w14:paraId="{para_id}"' if para_id else ""
    return (
        f'<w:comment w:id="{cid}" w:author="{author}" w:date="2024-01-01T00:00:00Z">'
        f"<w:p{para_id_attr}><w:r><w:t>{text}</w:t></w:r></w:p>"
        f"</w:comment>"
    )


def _doc_xml(body: str) -> bytes:
    return (f'<w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>').encode()


def _para(text: str) -> str:
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


def _make_docx(document_xml: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", document_xml)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _parse_comments
# ---------------------------------------------------------------------------


class TestParseComments:
    def test_empty_returns_empty(self):
        comments, para_map = _parse_comments(_comments_xml(""))
        assert comments == {}
        assert para_map == {}

    def test_single_comment(self):
        xml = _comments_xml(_comment_el("1", "Alice", "Great point"))
        comments, _ = _parse_comments(xml)
        assert "1" in comments
        c = comments["1"]
        assert c.author == "Alice"
        assert c.text == "Great point"
        assert c.id == "1"

    def test_multiple_comments(self):
        xml = _comments_xml(
            _comment_el("1", "Alice", "First") + _comment_el("2", "Bob", "Second")
        )
        comments, _ = _parse_comments(xml)
        assert len(comments) == 2
        assert comments["2"].author == "Bob"

    def test_para_id_mapped(self):
        xml = _comments_xml(_comment_el("1", "Alice", "text", para_id="AABB0011"))
        _, para_map = _parse_comments(xml)
        assert "AABB0011" in para_map
        assert para_map["AABB0011"] == "1"

    def test_default_resolved_is_false(self):
        xml = _comments_xml(_comment_el("1", "Alice", "text"))
        comments, _ = _parse_comments(xml)
        assert comments["1"].resolved is False

    def test_comment_missing_id_skipped(self):
        xml = _comments_xml(
            f'<w:comment w:author="Alice" w:date="2024-01-01T00:00:00Z">'
            f"<w:p><w:r><w:t>no id</w:t></w:r></w:p>"
            f"</w:comment>"
        )
        comments, _ = _parse_comments(xml)
        assert len(comments) == 0


# ---------------------------------------------------------------------------
# _build_tree
# ---------------------------------------------------------------------------


class TestBuildTree:
    def _make(self, cid: str, parent_id: str | None = None) -> Comment:
        c = Comment(id=cid, author="A", date="2024-01-01", text="t")
        c.parent_id = parent_id
        return c

    def test_flat_list_all_top_level(self):
        comments = {str(i): self._make(str(i)) for i in range(3)}
        top = _build_tree(comments)
        assert len(top) == 3

    def test_reply_nested_under_parent(self):
        parent = self._make("1")
        reply = self._make("2", parent_id="1")
        comments = {"1": parent, "2": reply}
        top = _build_tree(comments)
        assert len(top) == 1
        assert len(top[0].replies) == 1
        assert top[0].replies[0].id == "2"

    def test_multiple_replies(self):
        parent = self._make("1")
        r1 = self._make("2", parent_id="1")
        r2 = self._make("3", parent_id="1")
        comments = {"1": parent, "2": r1, "3": r2}
        top = _build_tree(comments)
        assert len(top) == 1
        assert len(top[0].replies) == 2

    def test_orphan_reply_becomes_top_level(self):
        # Reply references a non-existent parent
        reply = self._make("2", parent_id="999")
        comments = {"2": reply}
        top = _build_tree(comments)
        assert len(top) == 1


# ---------------------------------------------------------------------------
# _parse_document_context
# ---------------------------------------------------------------------------


class TestParseDocumentContext:
    def test_empty_document(self):
        ctx = _parse_document_context(_doc_xml(""))
        assert ctx == {}

    def test_selected_text_extracted(self):
        xml = _doc_xml(
            "<w:p>"
            '<w:r><w:t xml:space="preserve">Before </w:t></w:r>'
            '<w:commentRangeStart w:id="1"/>'
            "<w:r><w:t>selected</w:t></w:r>"
            '<w:commentRangeEnd w:id="1"/>'
            "<w:r><w:t> after</w:t></w:r>"
            "</w:p>"
        )
        ctx = _parse_document_context(xml)
        assert "1" in ctx
        assert ctx["1"].selected_text == "selected"

    def test_para_idx_zero_for_first_para(self):
        xml = _doc_xml(
            "<w:p>"
            '<w:commentRangeStart w:id="1"/>'
            "<w:r><w:t>text</w:t></w:r>"
            '<w:commentRangeEnd w:id="1"/>'
            "</w:p>"
        )
        ctx = _parse_document_context(xml)
        assert ctx["1"].start_para_idx == 0
        assert ctx["1"].end_para_idx == 0

    def test_para_idx_for_second_para(self):
        xml = _doc_xml(
            _para("first paragraph") + "<w:p>"
            '<w:commentRangeStart w:id="1"/>'
            "<w:r><w:t>anchored</w:t></w:r>"
            '<w:commentRangeEnd w:id="1"/>'
            "</w:p>"
        )
        ctx = _parse_document_context(xml)
        assert ctx["1"].start_para_idx == 1

    def test_char_span_correct(self):
        # Para text: "Before selected after"
        # "Before " = 7 chars → selected starts at 7, ends at 15
        xml = _doc_xml(
            "<w:p>"
            '<w:r><w:t xml:space="preserve">Before </w:t></w:r>'
            '<w:commentRangeStart w:id="1"/>'
            "<w:r><w:t>selected</w:t></w:r>"
            '<w:commentRangeEnd w:id="1"/>'
            "</w:p>"
        )
        ctx = _parse_document_context(xml)
        span = ctx["1"].selected_span
        assert span.start == 7
        assert span.end == 15


# ---------------------------------------------------------------------------
# extract_paragraphs
# ---------------------------------------------------------------------------


class TestExtractParagraphs:
    def test_empty_document(self):
        docx = _make_docx(f'<w:document xmlns:w="{W}"><w:body></w:body></w:document>')
        result = extract_paragraphs(io.BytesIO(docx))
        assert isinstance(result, DocumentParagraphs)
        assert result.paragraphs == []
        assert result.moved_from == {}

    def test_plain_paragraphs(self):
        docx = _make_docx(
            f'<w:document xmlns:w="{W}"><w:body>'
            "<w:p><w:r><w:t>First</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>Second</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        result = extract_paragraphs(io.BytesIO(docx))
        assert result.paragraphs == ["First", "Second"]

    def test_movefrom_excluded_from_paragraphs(self):
        W_NS = f'xmlns:w="{W}"'
        docx = _make_docx(
            f"<w:document {W_NS}><w:body>"
            f'<w:moveFrom w:id="1" w:author="A" w:date="2024-01-01T00:00:00Z">'
            "<w:p><w:r><w:t>moved away</w:t></w:r></w:p>"
            "</w:moveFrom>"
            "<w:p><w:r><w:t>kept</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        result = extract_paragraphs(io.BytesIO(docx))
        assert "kept" in result.paragraphs
        assert "moved away" not in result.paragraphs

    def test_movefrom_in_moved_from_dict(self):
        W_NS = f'xmlns:w="{W}"'
        docx = _make_docx(
            f"<w:document {W_NS}><w:body>"
            f'<w:moveFrom w:id="1" w:author="A" w:date="2024-01-01T00:00:00Z">'
            "<w:p><w:r><w:t>moved away</w:t></w:r></w:p>"
            "</w:moveFrom>"
            "</w:body></w:document>"
        )
        result = extract_paragraphs(io.BytesIO(docx))
        assert len(result.moved_from) == 1
        assert "moved away" in result.moved_from.values()
