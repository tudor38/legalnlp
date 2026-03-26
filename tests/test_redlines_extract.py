"""
Tests for the redline and move XML parsers in src/redlines/extract.py.

These test the private parsing functions directly with minimal XML fixtures,
covering the most important correctness properties:
  - insertion and deletion extraction
  - character position tracking
  - moveFrom paragraphs are excluded from para_idx counting
  - move pair matching (text equality check, missing-partner drop)
"""

from src.redlines.extract import _parse_moves, _parse_redlines

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _xml(body: str) -> bytes:
    return (
        f'<w:document xmlns:w="{W}">'
        f"<w:body>{body}</w:body>"
        f"</w:document>"
    ).encode()


def _p(*runs: str) -> str:
    """Build a <w:p> with plain text runs."""
    inner = "".join(f"<w:r><w:t>{r}</w:t></w:r>" for r in runs)
    return f"<w:p>{inner}</w:p>"


def _ins(rid: str, author: str, text: str) -> str:
    return (
        f'<w:ins w:id="{rid}" w:author="{author}" w:date="2024-01-01T00:00:00Z">'
        f"<w:r><w:t>{text}</w:t></w:r>"
        f"</w:ins>"
    )


def _del(rid: str, author: str, text: str) -> str:
    return (
        f'<w:del w:id="{rid}" w:author="{author}" w:date="2024-01-01T00:00:00Z">'
        f"<w:r><w:delText>{text}</w:delText></w:r>"
        f"</w:del>"
    )


def _move_from(mid: str, author: str, text: str) -> str:
    return (
        f'<w:moveFrom w:id="{mid}" w:author="{author}" w:date="2024-01-01T00:00:00Z">'
        f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"
        f"</w:moveFrom>"
    )


def _move_to(mid: str, text: str) -> str:
    return (
        f'<w:moveTo w:id="{mid}" w:author="Alice" w:date="2024-01-01T00:00:00Z">'
        f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"
        f"</w:moveTo>"
    )


# ---------------------------------------------------------------------------
# Redlines
# ---------------------------------------------------------------------------


class TestParseRedlines:
    def test_empty_document(self):
        xml = _xml("")
        assert _parse_redlines(xml) == []

    def test_single_insertion(self):
        xml = _xml(f"<w:p>{_ins('1', 'Alice', 'hello')}</w:p>")
        redlines = _parse_redlines(xml)
        assert len(redlines) == 1
        r = redlines[0]
        assert r.kind == "insertion"
        assert r.text == "hello"
        assert r.author == "Alice"
        assert r.id == "1"

    def test_single_deletion(self):
        xml = _xml(f"<w:p>{_del('2', 'Bob', 'removed')}</w:p>")
        redlines = _parse_redlines(xml)
        assert len(redlines) == 1
        r = redlines[0]
        assert r.kind == "deletion"
        assert r.text == "removed"
        assert r.author == "Bob"

    def test_char_positions_after_leading_text(self):
        # Para: "Before " (7 chars) + insertion "world" (5 chars)
        xml = _xml(
            f"<w:p>"
            f'<w:r><w:t xml:space="preserve">Before </w:t></w:r>'
            f"{_ins('1', 'Alice', 'world')}"
            f"</w:p>"
        )
        redlines = _parse_redlines(xml)
        assert len(redlines) == 1
        r = redlines[0]
        assert r.char_start == 7
        assert r.char_end == 12

    def test_para_idx_assigned(self):
        # Two paragraphs; insertion is in the second one.
        xml = _xml(
            _p("first paragraph")
            + f"<w:p>{_ins('1', 'Alice', 'inserted')}</w:p>"
        )
        redlines = _parse_redlines(xml)
        assert len(redlines) == 1
        assert redlines[0].context.para_idx == 1

    def test_movefrom_paragraphs_excluded_from_para_idx(self):
        # moveFrom para comes first in XML but must not increment para_idx.
        # The insertion is in the first non-moveFrom paragraph → para_idx 0.
        xml = _xml(
            _move_from("1", "Alice", "moved text")
            + f"<w:p>{_ins('2', 'Bob', 'inserted')}</w:p>"
        )
        redlines = _parse_redlines(xml)
        assert len(redlines) == 1
        assert redlines[0].context.para_idx == 0

    def test_multiple_redlines_same_para(self):
        xml = _xml(
            f"<w:p>"
            f"{_ins('1', 'Alice', 'foo')}"
            f"{_del('2', 'Alice', 'bar')}"
            f"</w:p>"
        )
        redlines = _parse_redlines(xml)
        assert len(redlines) == 2
        kinds = {r.kind for r in redlines}
        assert kinds == {"insertion", "deletion"}


# ---------------------------------------------------------------------------
# Moves
# ---------------------------------------------------------------------------


class TestParseMoves:
    def test_empty_document(self):
        assert _parse_moves(_xml("")) == []

    def test_matched_move_pair(self):
        xml = _xml(_move_from("1", "Alice", "moved text") + _move_to("1", "moved text"))
        moves = _parse_moves(xml)
        assert len(moves) == 1
        m = moves[0]
        assert m.text == "moved text"
        assert m.author == "Alice"
        assert m.id == "1"

    def test_mismatched_text_dropped(self):
        # from_text != to_text → pair must be dropped
        xml = _xml(_move_from("1", "Alice", "original") + _move_to("1", "different"))
        assert _parse_moves(xml) == []

    def test_missing_moveto_dropped(self):
        xml = _xml(_move_from("1", "Alice", "orphan"))
        assert _parse_moves(xml) == []

    def test_missing_movefrom_dropped(self):
        xml = _xml(_move_to("1", "orphan"))
        assert _parse_moves(xml) == []

    def test_move_indices(self):
        # Layout: [moveFrom para] [regular para] [moveTo para]
        # from_para_idx = 0 (first xml para)
        # to_para_idx   = 1 (second final-doc para; regular para is 0)
        xml = _xml(
            _move_from("1", "Alice", "moved")
            + _p("regular")
            + _move_to("1", "moved")
        )
        moves = _parse_moves(xml)
        assert len(moves) == 1
        assert moves[0].from_para_idx == 0
        assert moves[0].to_para_idx == 1
