"""Lexical retrieval with TF-IDF vectors and cosine similarity.

How it works:
    1. Fit a TF-IDF vectorizer on every chunk. Each chunk becomes a sparse
       vector with one dimension per vocabulary term (word or two-word phrase).
    2. Transform the question with the SAME fitted vectorizer, so it lives in
       the same vector space as the chunks.
    3. Cosine similarity between the question vector and every chunk vector
       gives a relevance score between 0 (no shared terms) and 1.
    4. Keep the best chunk per document, drop scores below the threshold,
       sort by score, return the top_k documents.

Limitation: this is LEXICAL matching. It only finds chunks that share words
with the question. It does not understand synonyms ("supporter" vs "donor")
or meaning. Semantic embeddings are planned for Phase 2.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import TypedDict

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import Settings, load_settings
from app.document_processor import Chunk, clean_text, process_documents
from app.knowledge_base import load_documents

MAX_QUESTION_LENGTH = 1000


class EmptyQuestionError(ValueError):
    """Raised when the question is empty or only whitespace."""


class RetrievedDocument(TypedDict):
    """One ranked search result. Plain dict so it is JSON-serializable."""

    id: str
    title: str
    category: str
    source_type: str
    chunk_id: str
    score: float
    matched_terms: list[str]
    content: str


def normalize_question(question: str) -> str:
    """Validate and clean a user question."""
    if not isinstance(question, str):
        raise TypeError(f"question must be a string, got {type(question).__name__}")
    cleaned = clean_text(question)
    if not cleaned:
        raise EmptyQuestionError("Question must not be empty.")
    if len(cleaned) > MAX_QUESTION_LENGTH:
        raise ValueError(f"Question is too long (max {MAX_QUESTION_LENGTH} characters).")
    return cleaned


def build_vectorizer() -> TfidfVectorizer:
    """Create the TF-IDF vectorizer used for both chunks and questions."""
    return TfidfVectorizer(
        lowercase=True,
        # Remove very common words ("how", "can", "we", "the") that carry no topic.
        stop_words="english",
        # Single words AND two-word phrases, e.g. "donor" and "donor retention".
        ngram_range=(1, 2),
        # Use 1 + log(tf) so a word repeated 10 times is not 10x more important.
        sublinear_tf=True,
    )


class TfidfRetriever:
    """Ranks knowledge-base chunks against a question using TF-IDF."""

    def __init__(self, chunks: Sequence[Chunk], min_similarity: float = 0.10) -> None:
        if not chunks:
            raise ValueError("Cannot build a retriever without chunks.")
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between 0 and 1.")

        self.chunks: list[Chunk] = list(chunks)
        self.min_similarity = min_similarity
        self.vectorizer = build_vectorizer()
        # Prefix each chunk with its document title ("contextual chunk header"),
        # so a chunk that never repeats the topic word still matches the topic.
        self.chunk_matrix: csr_matrix = self.vectorizer.fit_transform(
            [f"{chunk.title}. {chunk.text}" for chunk in self.chunks]
        )
        self._feature_names: np.ndarray = self.vectorizer.get_feature_names_out()

    @property
    def vocabulary_size(self) -> int:
        return len(self._feature_names)

    def vectorize_question(self, question: str) -> csr_matrix:
        """Turn a question into a TF-IDF vector in the chunks' vector space."""
        return self.vectorizer.transform([normalize_question(question)])

    def score_chunks(self, question: str) -> np.ndarray:
        """Cosine similarity between the question and every chunk (same order)."""
        question_vector = self.vectorize_question(question)
        return cosine_similarity(question_vector, self.chunk_matrix).ravel()

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
        min_similarity: float | None = None,
    ) -> list[RetrievedDocument]:
        """Return up to `top_k` relevant documents, best first.

        Each document is represented by its best-matching chunk. Chunks with a
        score of 0 or below the threshold are never returned.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        threshold = self.min_similarity if min_similarity is None else min_similarity

        question_vector = self.vectorize_question(question)
        scores = cosine_similarity(question_vector, self.chunk_matrix).ravel()

        best_by_document: dict[str, tuple[float, int]] = {}
        for row, score in enumerate(scores):
            if score <= 0.0 or score < threshold:
                continue
            document_id = self.chunks[row].document_id
            previous = best_by_document.get(document_id)
            if previous is None or score > previous[0]:
                best_by_document[document_id] = (float(score), row)

        # Highest score first; document ID breaks ties so output is deterministic.
        ranked = sorted(
            best_by_document.values(),
            key=lambda item: (-item[0], self.chunks[item[1]].document_id),
        )
        return [self._to_result(row, score, question_vector) for score, row in ranked[:top_k]]

    def matched_terms(self, question_vector: csr_matrix, row: int, limit: int = 5) -> list[str]:
        """Terms shared by question and chunk, largest contribution first.

        Both vectors are L2-normalized, so the element-wise products sum to the
        cosine similarity. This makes every score explainable.
        """
        contributions = question_vector.multiply(self.chunk_matrix[row]).tocoo()
        pairs = sorted(
            zip(contributions.col, contributions.data, strict=True), key=lambda pair: -pair[1]
        )
        return [str(self._feature_names[column]) for column, _ in pairs[:limit]]

    def _to_result(self, row: int, score: float, question_vector: csr_matrix) -> RetrievedDocument:
        chunk = self.chunks[row]
        return RetrievedDocument(
            id=chunk.document_id,
            title=chunk.title,
            category=chunk.category,
            source_type=chunk.source_type,
            chunk_id=chunk.chunk_id,
            score=round(score, 4),
            matched_terms=self.matched_terms(question_vector, row),
            content=chunk.text,
        )


def build_retriever(settings: Settings) -> TfidfRetriever:
    """Load documents, chunk them, and fit a retriever."""
    documents = load_documents(settings.documents_dir)
    chunks = process_documents(
        documents, settings.chunk_max_words, settings.chunk_overlap_sentences
    )
    return TfidfRetriever(chunks, min_similarity=settings.min_similarity)


@lru_cache(maxsize=1)
def get_default_retriever() -> TfidfRetriever:
    """Retriever built from environment settings, created once and reused."""
    return build_retriever(load_settings())


def retrieve_documents(question: str, top_k: int = 3) -> list[RetrievedDocument]:
    """Convenience function: search the default knowledge base."""
    return get_default_retriever().retrieve(question, top_k=top_k)
