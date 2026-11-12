"""Run the required practical questions plus off-topic and edge-case questions.

Usage (from the project root):
    python examples/sample_questions.py
    python examples/sample_questions.py --hide-prompt
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python examples/sample_questions.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.answer_service import AnswerService  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.formatter import format_result, separator  # noqa: E402
from app.retriever import build_retriever  # noqa: E402

REQUIRED_QUESTIONS = [
    "How can we improve donor retention?",
    "How can we measure campaign performance?",
    "What is donor segmentation?",
    "How can we protect personal information?",
    "What are effective donor acquisition strategies?",
]

# The system should return NO documents for these.
UNRELATED_QUESTIONS = [
    "What is the best recipe for chocolate cake?",
    "Who won the football world cup?",
]

# Known weaknesses of lexical retrieval, kept here on purpose for learning.
LIMITATION_QUESTIONS = [
    # Shares words ("improve", "performance") but is not about fundraising.
    "How do I improve my football team performance?",
    # Relevant to Data Privacy, but uses synonyms instead of the document's words.
    "How can we keep supporter details safe?",
]


def main() -> None:
    show_prompt = "--hide-prompt" not in sys.argv
    settings = load_settings()
    service = AnswerService(build_retriever(settings), top_k=settings.top_k)

    sections = [
        ("REQUIRED PRACTICAL QUESTIONS", REQUIRED_QUESTIONS),
        ("UNRELATED QUESTIONS (expect no documents)", UNRELATED_QUESTIONS),
        ("LEXICAL RETRIEVAL LIMITATIONS (see README)", LIMITATION_QUESTIONS),
    ]
    for heading, questions in sections:
        print(f"\n{separator()}\n{heading}\n{separator()}")
        for question in questions:
            result = service.answer(question)
            # Only show the long prompt for the required questions.
            print(
                format_result(
                    result,
                    settings.min_similarity,
                    show_prompt=show_prompt and questions is REQUIRED_QUESTIONS,
                )
            )
            print(separator())


if __name__ == "__main__":
    main()
