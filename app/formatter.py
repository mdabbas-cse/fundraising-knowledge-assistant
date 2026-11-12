"""Plain-text rendering of an AnswerResult for the CLI and example scripts."""

from __future__ import annotations

from app.answer_service import AnswerResult, AnswerStatus

_RULE = "=" * 72


def format_result(result: AnswerResult, min_similarity: float, show_prompt: bool = True) -> str:
    lines: list[str] = [f"Question: {result.question}", ""]

    if result.status is AnswerStatus.INVALID_QUESTION:
        lines.append(f"Invalid question: {result.message}")
        return "\n".join(lines)

    lines += ["Relevant Documents:", ""]
    if not result.retrieved_documents:
        lines += [
            f"  None. Every similarity score was below the threshold ({min_similarity:.2f}).",
            "",
            result.message,
            "No prompt was built, so no LLM call would be made.",
        ]
        return "\n".join(lines)

    for rank, document in enumerate(result.retrieved_documents, start=1):
        lines += [
            f"{rank}. {document['title']}",
            f"   ID: {document['id']}",
            f"   Similarity: {document['score']:.4f}",
            f"   Best chunk: {document['chunk_id']}",
            f"   Matched terms: {', '.join(document['matched_terms'])}",
        ]
    lines.append("")

    if show_prompt and result.prompt:
        lines += ["Generated LLM Prompt:", "-" * 72, result.prompt, "-" * 72, ""]

    if result.answer is not None:
        lines += ["Answer:", result.answer, ""]
        if result.unsupported_citations:
            lines += [
                "WARNING: answer cites documents that were not retrieved: "
                + ", ".join(result.unsupported_citations),
                "",
            ]

    lines.append("Sources:")
    lines += [f"- {source.id}: {source.title}" for source in result.sources]
    lines += ["", f"Status: {result.status.value} - {result.message}"]
    return "\n".join(lines)


def separator() -> str:
    return _RULE
