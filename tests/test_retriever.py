"""Tests for TF-IDF vectorization, cosine similarity and ranking."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.document_processor import Chunk
from app.knowledge_base import Document
from app.retriever import (
    MAX_QUESTION_LENGTH,
    EmptyQuestionError,
    TfidfRetriever,
    retrieve_documents,
)
from tests.conftest import REQUIRED_QUESTIONS, UNRELATED_QUESTIONS

# --- TF-IDF vectorization ----------------------------------------------------


def test_chunk_matrix_has_one_row_per_chunk(project_retriever: TfidfRetriever) -> None:
    rows, columns = project_retriever.chunk_matrix.shape
    assert rows == len(project_retriever.chunks)
    assert columns == project_retriever.vocabulary_size


def test_vocabulary_contains_words_and_bigrams_but_not_stop_words(
    project_retriever: TfidfRetriever,
) -> None:
    vocabulary = project_retriever.vectorizer.vocabulary_
    assert "retention" in vocabulary
    assert "donor retention" in vocabulary  # bigram
    assert "the" not in vocabulary and "how" not in vocabulary  # stop words


def test_chunk_vectors_are_l2_normalized(project_retriever: TfidfRetriever) -> None:
    matrix = project_retriever.chunk_matrix
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1))
    assert np.allclose(norms, 1.0)


def test_question_vector_uses_same_vocabulary(project_retriever: TfidfRetriever) -> None:
    vector = project_retriever.vectorize_question("How can we improve donor retention?")
    assert vector.shape == (1, project_retriever.vocabulary_size)
    assert vector.nnz > 0


def test_unknown_words_produce_an_empty_vector(project_retriever: TfidfRetriever) -> None:
    assert project_retriever.vectorize_question("chocolate cake recipe").nnz == 0


# --- Similarity calculation --------------------------------------------------


def test_cosine_similarity_on_a_tiny_example() -> None:
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(["donor retention", "data privacy"])
    scores = cosine_similarity(vectorizer.transform(["donor retention"]), matrix).ravel()
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_scores_are_between_zero_and_one(project_retriever: TfidfRetriever) -> None:
    scores = project_retriever.score_chunks("How can we improve donor retention?")
    assert scores.shape == (len(project_retriever.chunks),)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0 + 1e-9)


def test_score_equals_dot_product_of_normalized_vectors(small_retriever: TfidfRetriever) -> None:
    question = "How do we keep donors giving?"
    question_vector = small_retriever.vectorize_question(question)
    expected = (small_retriever.chunk_matrix @ question_vector.T).toarray().ravel()
    assert np.allclose(small_retriever.score_chunks(question), expected)


def test_chunk_text_is_most_similar_to_itself(small_retriever: TfidfRetriever) -> None:
    target: Chunk = small_retriever.chunks[1]
    scores = small_retriever.score_chunks(f"{target.title}. {target.text}")
    assert int(np.argmax(scores)) == 1
    assert scores[1] == pytest.approx(1.0)


def test_scores_are_calculated_not_hardcoded(project_retriever: TfidfRetriever) -> None:
    first = project_retriever.retrieve("How can we improve donor retention?")[0]["score"]
    second = project_retriever.retrieve("What is donor segmentation?")[0]["score"]
    assert first != second


# --- Ranking and top_k ---------------------------------------------------------


@pytest.mark.parametrize(("question", "expected_id"), REQUIRED_QUESTIONS.items())
def test_required_questions_rank_expected_document_first(
    project_retriever: TfidfRetriever, question: str, expected_id: str
) -> None:
    results = project_retriever.retrieve(question, top_k=3)
    assert results, f"No documents retrieved for: {question}"
    assert results[0]["id"] == expected_id


def test_results_are_sorted_by_descending_score(project_retriever: TfidfRetriever) -> None:
    results = project_retriever.retrieve("donor data campaign", top_k=6, min_similarity=0.0)
    scores = [result["score"] for result in results]
    assert len(results) > 1
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_top_k_limits_number_of_results(project_retriever: TfidfRetriever, top_k: int) -> None:
    results = project_retriever.retrieve("donor data campaign", top_k=top_k, min_similarity=0.0)
    assert len(results) == top_k


def test_each_document_appears_at_most_once(project_retriever: TfidfRetriever) -> None:
    results = project_retriever.retrieve("donor", top_k=10, min_similarity=0.0)
    ids = [result["id"] for result in results]
    assert len(ids) == len(set(ids))


def test_invalid_top_k_raises(project_retriever: TfidfRetriever) -> None:
    with pytest.raises(ValueError):
        project_retriever.retrieve("donor retention", top_k=0)


# --- Similarity threshold --------------------------------------------------------


def test_all_results_meet_the_threshold(project_retriever: TfidfRetriever) -> None:
    results = project_retriever.retrieve("donor data campaign", top_k=6, min_similarity=0.05)
    assert all(result["score"] >= 0.05 for result in results)


def test_default_threshold_filters_weak_partial_matches(project_retriever: TfidfRetriever) -> None:
    question = "How can we improve donor retention?"
    unfiltered = project_retriever.retrieve(question, top_k=3, min_similarity=0.0)
    filtered = project_retriever.retrieve(question, top_k=3)
    assert len(unfiltered) == 3
    assert [result["id"] for result in filtered] == ["DOC001"]


def test_very_high_threshold_returns_nothing(project_retriever: TfidfRetriever) -> None:
    assert project_retriever.retrieve("donor retention", min_similarity=0.99) == []


def test_zero_threshold_still_ignores_zero_scores(small_retriever: TfidfRetriever) -> None:
    results = small_retriever.retrieve("encrypt personal information", top_k=3, min_similarity=0.0)
    assert [result["id"] for result in results] == ["T002"]


# --- Empty questions and no matches --------------------------------------------


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_raises(project_retriever: TfidfRetriever, question: str) -> None:
    with pytest.raises(EmptyQuestionError):
        project_retriever.retrieve(question)


def test_non_string_question_raises(project_retriever: TfidfRetriever) -> None:
    with pytest.raises(TypeError):
        project_retriever.retrieve(None)  # type: ignore[arg-type]


def test_too_long_question_raises(project_retriever: TfidfRetriever) -> None:
    with pytest.raises(ValueError, match="too long"):
        project_retriever.retrieve("donor " * MAX_QUESTION_LENGTH)


@pytest.mark.parametrize("question", UNRELATED_QUESTIONS)
def test_unrelated_question_returns_no_documents(
    project_retriever: TfidfRetriever, question: str
) -> None:
    assert project_retriever.retrieve(question) == []


def test_stop_word_only_question_returns_no_documents(project_retriever: TfidfRetriever) -> None:
    assert project_retriever.retrieve("How can we do it?") == []


# --- Metadata and source IDs -------------------------------------------------------


def test_results_preserve_source_metadata(
    project_retriever: TfidfRetriever, project_documents: list[Document]
) -> None:
    titles = {document.id: document.title for document in project_documents}
    results = project_retriever.retrieve("donor data campaign", top_k=6, min_similarity=0.0)
    for result in results:
        assert set(result) == {
            "id",
            "title",
            "category",
            "source_type",
            "chunk_id",
            "score",
            "matched_terms",
            "content",
        }
        assert result["title"] == titles[result["id"]]
        assert result["chunk_id"].startswith(f"{result['id']}-C")
        assert result["source_type"] == "synthetic_training_document"
        assert result["content"]


def test_matched_terms_explain_the_match(project_retriever: TfidfRetriever) -> None:
    result = project_retriever.retrieve("How can we improve donor retention?")[0]
    assert "retention" in result["matched_terms"]


def test_empty_chunk_list_raises() -> None:
    with pytest.raises(ValueError):
        TfidfRetriever([])


def test_retrieve_documents_convenience_function() -> None:
    results = retrieve_documents("How can we improve donor retention?", top_k=3)
    assert results[0]["id"] == "DOC001"
    assert isinstance(results[0]["score"], float)
