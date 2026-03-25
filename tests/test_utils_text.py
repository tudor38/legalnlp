import numpy as np
import pytest

from src.utils.text import bm25_scores, tokenize


class TestTokenize:
    def test_lowercases(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_splits_on_punctuation(self):
        assert tokenize("one, two; three.") == ["one", "two", "three"]

    def test_empty_string(self):
        assert tokenize("") == []

    def test_numbers_included(self):
        assert "42" in tokenize("section 42 applies")


class TestBm25Scores:
    DOCS = [
        "the cat sat on the mat",
        "the dog barked at the cat",
        "a quick brown fox",
    ]

    def test_returns_correct_shape(self):
        scores = bm25_scores(self.DOCS, "cat")
        assert scores.shape == (3,)

    def test_relevant_docs_score_higher(self):
        scores = bm25_scores(self.DOCS, "cat")
        # docs 0 and 1 contain "cat"; doc 2 does not
        assert scores[0] > scores[2]
        assert scores[1] > scores[2]

    def test_empty_query_returns_zeros(self):
        scores = bm25_scores(self.DOCS, "")
        np.testing.assert_array_equal(scores, np.zeros(3))

    def test_query_not_in_any_doc_returns_zeros(self):
        scores = bm25_scores(self.DOCS, "elephant")
        np.testing.assert_array_equal(scores, np.zeros(3))

    def test_single_doc(self):
        scores = bm25_scores(["only document here"], "document")
        assert scores[0] > 0

    def test_empty_docs(self):
        scores = bm25_scores([], "cat")
        assert scores.shape == (0,)
