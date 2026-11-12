"""Tests for text cleaning and chunk creation."""

from __future__ import annotations

from itertools import pairwise

import pytest

from app.document_processor import (
    chunk_text,
    clean_text,
    process_document,
    process_documents,
    split_into_sentences,
)
from app.knowledge_base import Document
from tests.conftest import make_document

# Ten sentences of exactly six words each: "Sentence 0 has exactly six words."
SENTENCES = [f"Sentence {i} has exactly six words." for i in range(10)]
LONG_TEXT = " ".join(SENTENCES)


# --- Text processing -------------------------------------------------------


def test_clean_text_collapses_whitespace_and_newlines() -> None:
    assert clean_text("  Donor \n\n retention\t matters.  ") == "Donor retention matters."


def test_clean_text_normalizes_typographic_characters() -> None:
    # Curly quotes, an en dash and a non-breaking space (written as escapes on purpose).
    raw = "\u201cDonor\u2019s gift\u201d \u2013 thank\u00a0you"
    assert clean_text(raw) == '"Donor\'s gift" - thank you'


def test_clean_text_preserves_case() -> None:
    assert clean_text("RFM Analysis") == "RFM Analysis"


def test_clean_text_of_blank_string_is_empty() -> None:
    assert clean_text(" \n\t ") == ""


def test_split_into_sentences() -> None:
    assert split_into_sentences("One. Two? Three!") == ["One.", "Two?", "Three!"]


# --- Chunk creation ---------------------------------------------------------


def test_short_text_becomes_single_chunk() -> None:
    assert chunk_text("Short text. Only two sentences.", max_words=50) == [
        "Short text. Only two sentences."
    ]


def test_empty_text_produces_no_chunks() -> None:
    assert chunk_text("   ", max_words=50) == []


def test_long_text_is_split_without_losing_sentences() -> None:
    chunks = chunk_text(LONG_TEXT, max_words=20, overlap_sentences=0)
    # 3 + 3 + 3 sentences, and the tiny final sentence is merged into the last chunk.
    assert len(chunks) == 3
    assert " ".join(chunks) == LONG_TEXT


def test_chunks_respect_soft_word_limit() -> None:
    max_words = 20
    for chunk in chunk_text(LONG_TEXT, max_words=max_words, overlap_sentences=1):
        assert len(chunk.split()) <= max_words * 1.5


def test_overlap_repeats_last_sentence_of_previous_chunk() -> None:
    chunks = chunk_text(LONG_TEXT, max_words=20, overlap_sentences=1)
    assert len(chunks) > 1
    for previous, following in pairwise(chunks):
        last_sentence = split_into_sentences(previous)[-1]
        assert following.startswith(last_sentence)


def test_tiny_final_chunk_is_merged_into_previous() -> None:
    chunks = chunk_text(LONG_TEXT, max_words=20, overlap_sentences=0)
    assert chunks[-1].endswith(SENTENCES[-1])
    assert len(split_into_sentences(chunks[-1])) == 4


def test_invalid_chunk_settings_raise() -> None:
    with pytest.raises(ValueError):
        chunk_text(LONG_TEXT, max_words=0)
    with pytest.raises(ValueError):
        chunk_text(LONG_TEXT, max_words=10, overlap_sentences=-1)


# --- Metadata preservation ---------------------------------------------------


def test_chunks_keep_document_metadata() -> None:
    document = make_document("DOC777", "Donor Retention", LONG_TEXT)
    chunks = process_document(document, max_words=20, overlap_sentences=1)
    assert len(chunks) > 1
    for index, chunk in enumerate(chunks, start=1):
        assert chunk.document_id == "DOC777"
        assert chunk.title == "Donor Retention"
        assert chunk.category == document.category
        assert chunk.source_type == "synthetic_training_document"
        assert chunk.chunk_index == index
        assert chunk.chunk_id == f"DOC777-C{index:02d}"


def test_every_project_document_produces_chunks(project_documents: list[Document]) -> None:
    chunks = process_documents(project_documents, max_words=80, overlap_sentences=1)
    assert {chunk.document_id for chunk in chunks} == {d.id for d in project_documents}
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert len(chunks) > len(project_documents)  # long documents were split
