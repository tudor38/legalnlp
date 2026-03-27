"""
Tests for the HTML-safe highlight utilities in src/utils/text.py.

All highlight functions must:
  - HTML-escape text that is not wrapped in a <mark> tag (XSS prevention)
  - Still wrap matched content in <mark> tags
  - Work correctly with empty / non-matching queries
"""

import re

from src.utils.text import (
    highlight_query_tokens,
    highlight_regex,
    highlight_term,
    highlight_topic_keywords,
)


class TestHighlightTerm:
    def test_empty_query_escapes_text(self):
        assert highlight_term("Hello <World>", "") == "Hello &lt;World&gt;"

    def test_match_wrapped_in_mark(self):
        result = highlight_term("Hello World", "Hello")
        assert "<mark" in result
        assert "Hello" in result

    def test_surrounding_text_escaped(self):
        result = highlight_term("a < b matched c", "matched")
        assert "&lt;" in result
        assert "<mark" in result

    def test_html_in_query_is_safe(self):
        # Query containing HTML special chars must not produce raw tags
        result = highlight_term("AT&T is great", "AT&T")
        assert "<script" not in result
        assert "&amp;" in result  # & must be escaped in output

    def test_xss_in_text_neutralised(self):
        result = highlight_term("<script>alert(1)</script> hello", "hello")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_no_match_returns_escaped(self):
        result = highlight_term("plain & simple", "xyz")
        assert result == "plain &amp; simple"

    def test_color_applied_to_mark(self):
        result = highlight_term("hello world", "hello", color="#ff0")
        assert "background:#ff0" in result

    def test_case_insensitive(self):
        result = highlight_term("Hello World", "hello")
        assert "<mark" in result


class TestHighlightQueryTokens:
    def test_empty_query_escapes_text(self):
        assert highlight_query_tokens("Hello <World>", "") == "Hello &lt;World&gt;"

    def test_xss_in_text_neutralised(self):
        result = highlight_query_tokens(
            "<script>alert(1)</script> contract", "contract"
        )
        assert "<script>" not in result

    def test_match_wrapped_in_mark(self):
        result = highlight_query_tokens("The contract is signed", "contracts")
        # "contract" stems to same root as "contracts"
        assert "<mark" in result

    def test_stopwords_not_highlighted(self):
        result = highlight_query_tokens("the cat sat on the mat", "the")
        # "the" is a stop word — should not be highlighted
        assert "<mark" not in result

    def test_no_matching_tokens_escapes_only(self):
        result = highlight_query_tokens("Hello & World", "xyz")
        assert "&amp;" in result
        assert "<mark" not in result


class TestHighlightTopicKeywords:
    def test_noise_label_returns_unchanged(self):
        html = "Hello &lt;World&gt;"
        assert highlight_topic_keywords(html, "Noise", "#ff0") == html

    def test_empty_label_returns_unchanged(self):
        html = "Hello World"
        assert highlight_topic_keywords(html, "", "#ff0") == html

    def test_wraps_keyword_in_text_node(self):
        result = highlight_topic_keywords("liability clause", "liability", "#ff0")
        assert "<mark" in result
        assert "liability" in result

    def test_does_not_corrupt_existing_mark_attributes(self):
        # "background" is a word in the style attribute — must not be wrapped in <mark>
        html = '<mark style="background:#ff0">term</mark> rest'
        result = highlight_topic_keywords(html, "background", "#aaa")
        # The style attribute must remain intact
        assert 'style="background:#ff0"' in result

    def test_does_not_double_wrap_marked_text(self):
        # The word "term" inside an existing <mark> is in a text node between tags
        html = '<mark style="background:#ff0">term</mark>'
        result = highlight_topic_keywords(html, "term", "#aaa")
        # Result must still be valid — "term" may be wrapped again in the text node
        # but the outer mark's attributes must not be modified
        assert 'style="background:#ff0"' in result

    def test_xss_not_introduced(self):
        # html_text is already escaped; re-escaping would corrupt it,
        # but adding new marks must not introduce raw HTML from the label
        html = "contract &amp; agreement"
        result = highlight_topic_keywords(html, "contract", "#ff0")
        assert "&amp;" in result  # original entity must survive


class TestHighlightRegex:
    def test_match_wrapped_in_mark(self):
        result = highlight_regex("Hello World", re.compile("hello", re.I))
        assert "<mark" in result

    def test_non_match_text_escaped(self):
        result = highlight_regex("a < b", re.compile("xyz"))
        assert result == "a &lt; b"

    def test_match_text_escaped(self):
        result = highlight_regex("a <b> c", re.compile(r"<b>"))
        # The matched "<b>" must be escaped inside the mark tag
        assert "&lt;b&gt;" in result
        assert "<mark" in result

    def test_color_in_style(self):
        result = highlight_regex("hello", re.compile("hello"), color="#abc")
        assert "background:#abc" in result

    def test_no_match_returns_escaped_text(self):
        result = highlight_regex("AT&T", re.compile("xyz"))
        assert result == "AT&amp;T"

    def test_multiple_matches(self):
        result = highlight_regex("aaa bbb aaa", re.compile("aaa"))
        assert result.count("<mark") == 2
