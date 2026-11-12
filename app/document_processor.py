"""Document processing: clean text and split documents into chunks.

Why chunk?
    * A long document covers several sub-topics. One vector for the whole
      document mixes them together and weakens the match for a specific question.
    * An LLM prompt has a limited size, so we only want to send the most
      relevant passages, not whole documents.

Chunking strategy (simple and explainable):
    1. Split the cleaned text into sentences.
    2. Pack whole sentences into a chunk until it would exceed `max_words`.
    3. Start the next chunk with the last `overlap_sentences` sentences so an
       idea that crosses a chunk boundary is not lost.
    4. If the final chunk is tiny, merge it into the previous chunk.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.knowledge_base import Document

# Typographic characters that commonly appear in copied text.
_CHARACTER_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",
}
_WHITESPACE = re.compile(r"\s+")
# A sentence ends with . ! or ? followed by whitespace.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
# A final chunk with fewer new words than this share of max_words is merged
# into the previous chunk.
MIN_FINAL_CHUNK_RATIO = 0.5


@dataclass(frozen=True)
class Chunk:
    """A piece of a document, with enough metadata to cite its source."""

    chunk_id: str
    document_id: str
    title: str
    category: str
    source_type: str
    chunk_index: int
    text: str


def clean_text(text: str) -> str:
    """Normalize unicode, replace typographic characters, collapse whitespace.

    Case is preserved because the text is also shown to users and the LLM;
    lowercasing for matching is done by the TF-IDF vectorizer.
    """
    normalized = unicodedata.normalize("NFKC", text)
    for original, replacement in _CHARACTER_REPLACEMENTS.items():
        normalized = normalized.replace(original, replacement)
    return _WHITESPACE.sub(" ", normalized).strip()


def split_into_sentences(text: str) -> list[str]:
    """Split cleaned text into sentences (simple rule-based splitter)."""
    return [sentence for sentence in _SENTENCE_BOUNDARY.split(text) if sentence]


def chunk_text(text: str, max_words: int, overlap_sentences: int = 1) -> list[str]:
    """Group sentences into chunks of roughly `max_words` words.

    `max_words` is a soft limit: a single sentence longer than the limit becomes
    its own chunk instead of being cut in the middle, and a tiny final chunk is
    merged into the previous one (that chunk can reach ~1.5x `max_words`).
    """
    if max_words < 1:
        raise ValueError("max_words must be >= 1")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")

    chunks: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    carried = 0  # how many sentences at the start of `current` are overlap

    for sentence in split_into_sentences(clean_text(text)):
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > max_words:
            chunks.append(current)
            current = current[-overlap_sentences:] if overlap_sentences else []
            carried = len(current)
            current_words = _count_words(current)
        current.append(sentence)
        current_words += sentence_words

    if current:
        new_sentences = current[carried:]
        # A tiny final chunk gets an unfairly high cosine score (few terms, so
        # each term weighs more) and gives the LLM little context. Merge it
        # into the previous chunk instead.
        if chunks and _count_words(new_sentences) < max_words * MIN_FINAL_CHUNK_RATIO:
            chunks[-1].extend(new_sentences)
        else:
            chunks.append(current)
    return [" ".join(sentences) for sentences in chunks]


def _count_words(sentences: list[str]) -> int:
    return sum(len(sentence.split()) for sentence in sentences)


def process_document(document: Document, max_words: int, overlap_sentences: int = 1) -> list[Chunk]:
    """Turn one document into chunks that keep the document's metadata."""
    pieces = chunk_text(document.content, max_words, overlap_sentences)
    return [
        Chunk(
            chunk_id=f"{document.id}-C{index:02d}",
            document_id=document.id,
            title=clean_text(document.title),
            category=document.category,
            source_type=document.source_type,
            chunk_index=index,
            text=piece,
        )
        for index, piece in enumerate(pieces, start=1)
    ]


def process_documents(
    documents: list[Document], max_words: int, overlap_sentences: int = 1
) -> list[Chunk]:
    """Chunk every document, preserving document order."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(process_document(document, max_words, overlap_sentences))
    return chunks
