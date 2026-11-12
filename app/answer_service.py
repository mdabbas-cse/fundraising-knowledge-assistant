"""Answer service: question -> retrieval -> prompt -> (optional LLM) -> result.

Phase 1: no LLM is configured, so the result contains the prompt and sources
but `answer` is None. We never pretend an AI answer exists.

Phase 2: pass any object with a `generate(prompt) -> str` method as
`llm_provider` (OpenAI, Anthropic, a local model...). No other code changes.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from app.prompt_builder import INSUFFICIENT_INFORMATION_MESSAGE, build_prompt
from app.retriever import RetrievedDocument, normalize_question

# Matches citations such as [DOC001] in a generated answer.
_CITATION_PATTERN = re.compile(r"\[([A-Z]{2,}\d+)\]")


class Retriever(Protocol):
    """Anything that can rank documents for a question (TF-IDF now, embeddings later)."""

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedDocument]: ...


class LLMProvider(Protocol):
    """Anything that can turn a prompt into text. Implemented in Phase 2."""

    def generate(self, prompt: str) -> str: ...


class LLMProviderError(RuntimeError):
    """Raised when the configured LLM provider fails."""


class AnswerStatus(StrEnum):
    PROMPT_READY = "prompt_ready"  # Phase 1: prompt built, no LLM configured.
    ANSWERED = "answered"  # Phase 2: LLM produced an answer.
    NO_RELEVANT_DOCUMENTS = "no_relevant_documents"
    INVALID_QUESTION = "invalid_question"


@dataclass(frozen=True)
class SourceReference:
    id: str
    title: str


@dataclass
class AnswerResult:
    question: str
    status: AnswerStatus
    message: str
    retrieved_documents: list[RetrievedDocument] = field(default_factory=list)
    sources: list[SourceReference] = field(default_factory=list)
    prompt: str | None = None
    answer: str | None = None
    # Document IDs cited by the LLM that were NOT in the retrieved context.
    unsupported_citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (useful for a FastAPI response)."""
        data = asdict(self)
        data["status"] = self.status.value
        return data


def extract_sources(retrieved_docs: list[RetrievedDocument]) -> list[SourceReference]:
    """Unique (id, title) pairs in ranking order."""
    seen: set[str] = set()
    sources: list[SourceReference] = []
    for document in retrieved_docs:
        if document["id"] not in seen:
            seen.add(document["id"])
            sources.append(SourceReference(id=document["id"], title=document["title"]))
    return sources


def find_unsupported_citations(answer: str, allowed_ids: set[str]) -> list[str]:
    """Return cited IDs that were not part of the context (possible hallucination)."""
    cited = dict.fromkeys(_CITATION_PATTERN.findall(answer))  # ordered, unique
    return [document_id for document_id in cited if document_id not in allowed_ids]


class AnswerService:
    """Coordinates retrieval, prompt construction and (optionally) generation."""

    def __init__(
        self,
        retriever: Retriever,
        llm_provider: LLMProvider | None = None,
        top_k: int = 3,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.top_k = top_k

    def answer(self, question: str, top_k: int | None = None) -> AnswerResult:
        try:
            cleaned_question = normalize_question(question)
        except (TypeError, ValueError) as exc:
            return AnswerResult(
                question=str(question),
                status=AnswerStatus.INVALID_QUESTION,
                message=str(exc),
            )

        effective_top_k = self.top_k if top_k is None else top_k
        retrieved = self.retriever.retrieve(cleaned_question, top_k=effective_top_k)
        if not retrieved:
            # Do not build a prompt or call an LLM: with no context, any answer
            # would come from the model's memory, not from our knowledge base.
            return AnswerResult(
                question=cleaned_question,
                status=AnswerStatus.NO_RELEVANT_DOCUMENTS,
                message=INSUFFICIENT_INFORMATION_MESSAGE,
            )

        prompt = build_prompt(cleaned_question, retrieved)
        sources = extract_sources(retrieved)

        if self.llm_provider is None:
            return AnswerResult(
                question=cleaned_question,
                status=AnswerStatus.PROMPT_READY,
                message=(
                    "Prompt built from the retrieved context. No LLM is configured "
                    "(Phase 1), so no answer was generated."
                ),
                retrieved_documents=retrieved,
                sources=sources,
                prompt=prompt,
            )

        try:
            answer = self.llm_provider.generate(prompt)
        except Exception as exc:  # Provider SDKs raise many different error types.
            raise LLMProviderError(f"LLM provider failed: {exc}") from exc

        return AnswerResult(
            question=cleaned_question,
            status=AnswerStatus.ANSWERED,
            message="Answer generated from the retrieved context.",
            retrieved_documents=retrieved,
            sources=sources,
            prompt=prompt,
            answer=answer,
            unsupported_citations=find_unsupported_citations(
                answer, {source.id for source in sources}
            ),
        )
