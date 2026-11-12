"""Tests for the answer service (orchestration + optional LLM injection)."""

from __future__ import annotations

import json

import pytest

from app.answer_service import (
    AnswerService,
    AnswerStatus,
    LLMProviderError,
    extract_sources,
    find_unsupported_citations,
)
from app.prompt_builder import INSUFFICIENT_INFORMATION_MESSAGE
from app.retriever import RetrievedDocument, TfidfRetriever


class FakeLLM:
    """Test double that records prompts and returns a canned answer."""

    def __init__(self, answer: str = "Thank donors quickly [DOC001].") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


class FailingLLM:
    def generate(self, prompt: str) -> str:
        raise TimeoutError("provider timed out")


class RecordingRetriever:
    """Wraps a real retriever and counts calls."""

    def __init__(self, inner: TfidfRetriever) -> None:
        self.inner = inner
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedDocument]:
        self.calls.append((question, top_k))
        return self.inner.retrieve(question, top_k=top_k)


@pytest.fixture
def service(project_retriever: TfidfRetriever) -> AnswerService:
    return AnswerService(project_retriever)


# --- Phase 1 behaviour (no LLM) ---------------------------------------------------


def test_relevant_question_returns_prompt_and_sources(service: AnswerService) -> None:
    result = service.answer("How can we improve donor retention?")
    assert result.status is AnswerStatus.PROMPT_READY
    assert result.answer is None  # no fake AI answer in Phase 1
    assert result.prompt is not None
    assert result.sources[0].id == "DOC001"
    assert result.sources[0].title == "Donor Retention"


def test_sources_match_retrieved_documents(service: AnswerService) -> None:
    result = service.answer("donor data campaign", top_k=3)
    assert [s.id for s in result.sources] == [d["id"] for d in result.retrieved_documents]
    for source in result.sources:
        assert f"ID: {source.id}" in (result.prompt or "")


def test_unrelated_question_returns_no_documents(service: AnswerService) -> None:
    result = service.answer("What is the best recipe for chocolate cake?")
    assert result.status is AnswerStatus.NO_RELEVANT_DOCUMENTS
    assert result.message == INSUFFICIENT_INFORMATION_MESSAGE
    assert result.retrieved_documents == []
    assert result.sources == []
    assert result.prompt is None


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_is_rejected_without_retrieval(
    project_retriever: TfidfRetriever, question: str
) -> None:
    retriever = RecordingRetriever(project_retriever)
    result = AnswerService(retriever).answer(question)
    assert result.status is AnswerStatus.INVALID_QUESTION
    assert retriever.calls == []


def test_top_k_override_is_passed_to_retriever(project_retriever: TfidfRetriever) -> None:
    retriever = RecordingRetriever(project_retriever)
    AnswerService(retriever, top_k=3).answer("donor retention", top_k=1)
    assert retriever.calls == [("donor retention", 1)]


def test_invalid_default_top_k_raises(project_retriever: TfidfRetriever) -> None:
    with pytest.raises(ValueError):
        AnswerService(project_retriever, top_k=0)


def test_result_is_json_serializable(service: AnswerService) -> None:
    data = service.answer("What is donor segmentation?").to_dict()
    encoded = json.loads(json.dumps(data))
    assert encoded["status"] == "prompt_ready"
    assert encoded["sources"][0] == {"id": "DOC005", "title": "Donor Segmentation"}


# --- Phase 2 readiness (injected LLM) ------------------------------------------------


def test_injected_llm_receives_prompt_and_answer_is_returned(
    project_retriever: TfidfRetriever,
) -> None:
    llm = FakeLLM()
    result = AnswerService(project_retriever, llm_provider=llm).answer(
        "How can we improve donor retention?"
    )
    assert result.status is AnswerStatus.ANSWERED
    assert result.answer == llm.answer
    assert llm.prompts == [result.prompt]
    assert result.unsupported_citations == []


def test_llm_is_not_called_when_nothing_is_retrieved(project_retriever: TfidfRetriever) -> None:
    llm = FakeLLM()
    AnswerService(project_retriever, llm_provider=llm).answer("Who won the football world cup?")
    assert llm.prompts == []


def test_citations_outside_the_context_are_flagged(project_retriever: TfidfRetriever) -> None:
    llm = FakeLLM("Thank donors [DOC001]. Also see [DOC999].")
    result = AnswerService(project_retriever, llm_provider=llm).answer(
        "How can we improve donor retention?"
    )
    assert result.unsupported_citations == ["DOC999"]


def test_provider_errors_are_wrapped(project_retriever: TfidfRetriever) -> None:
    service = AnswerService(project_retriever, llm_provider=FailingLLM())
    with pytest.raises(LLMProviderError, match="timed out"):
        service.answer("How can we improve donor retention?")


# --- Helpers ----------------------------------------------------------------------------


def test_extract_sources_removes_duplicates_and_keeps_order() -> None:
    docs = [
        {"id": "DOC002", "title": "Donor Acquisition"},
        {"id": "DOC001", "title": "Donor Retention"},
        {"id": "DOC002", "title": "Donor Acquisition"},
    ]
    sources = extract_sources(docs)  # type: ignore[arg-type]
    assert [(s.id, s.title) for s in sources] == [
        ("DOC002", "Donor Acquisition"),
        ("DOC001", "Donor Retention"),
    ]


def test_find_unsupported_citations() -> None:
    answer = "A [DOC001]. B [DOC003]. C [DOC003]. D [1]."
    assert find_unsupported_citations(answer, {"DOC001"}) == ["DOC003"]
