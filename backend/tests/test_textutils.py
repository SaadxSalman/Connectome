"""Lexical utility circuit tests."""

from app.textutils import (
    lexical_overlap, sentence_split, slugify, tokenize, truncate,
)


def test_tokenize_drops_stopwords_and_single_chars():
    assert "the" not in tokenize("The quick brown fox")
    assert tokenize("T4 neurons fire 120 spikes") == ["t4", "neurons", "fire", "120", "spikes"]


def test_sentence_split_keeps_abbrev_ish_boundaries_sane():
    sents = sentence_split("Motion flows. T4 neurons fire. Then what?")
    assert sents == ["Motion flows.", "T4 neurons fire.", "Then what?"]


def test_lexical_overlap_bounds():
    assert lexical_overlap(set(), "anything") == 0.0
    assert lexical_overlap({"motion", "detector"}, "a motion detector circuit") > 0.0
    assert lexical_overlap({"zzzz"}, "nothing matches") == 0.0


def test_slugify_and_truncate():
    assert slugify("Optic Lobe Circuits!") == "optic-lobe-circuits"
    assert truncate("x" * 50, 10).endswith("…")
    assert len(truncate("x" * 50, 10)) == 10
