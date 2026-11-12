"""Command-line interface for the Fundraising Knowledge Assistant.

Usage:
    python main.py                                   # interactive mode
    python main.py -q "What is donor segmentation?"  # one-off question(s)
    python main.py --top-k 2 --hide-prompt
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from app.answer_service import AnswerService
from app.config import PROJECT_ROOT, ConfigError, Settings, load_settings
from app.formatter import format_result, separator
from app.knowledge_base import KnowledgeBaseError
from app.retriever import build_retriever

EXIT_COMMANDS = {"exit", "quit", "q"}
BANNER = (
    "Fundraising Knowledge Assistant (Phase 1: TF-IDF retrieval baseline)\n"
    "All knowledge documents are SYNTHETIC training data.\n"
    "Type 'exit' or press Ctrl+D to quit."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fundraising Knowledge Assistant (RAG, Phase 1)")
    parser.add_argument(
        "-q",
        "--question",
        action="append",
        help="Ask a question and exit. Repeat the flag to ask several questions.",
    )
    parser.add_argument("--top-k", type=int, help="Maximum number of documents to return.")
    parser.add_argument("--hide-prompt", action="store_true", help="Do not print the LLM prompt.")
    return parser.parse_args(argv)


def create_service(settings: Settings, top_k: int | None) -> AnswerService:
    retriever = build_retriever(settings)
    return AnswerService(retriever, llm_provider=None, top_k=top_k or settings.top_k)


def ask(service: AnswerService, settings: Settings, question: str, show_prompt: bool) -> None:
    result = service.answer(question)
    print(format_result(result, settings.min_similarity, show_prompt=show_prompt))
    print(separator())


def run_interactive(service: AnswerService, settings: Settings, show_prompt: bool) -> None:
    print(BANNER)
    while True:
        try:
            question = input("\nAsk your question:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if question.lower() in EXIT_COMMANDS:
            print("Goodbye.")
            return
        print()
        ask(service, settings, question, show_prompt)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Values already set in the shell take priority over the .env file.
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    try:
        settings = load_settings()
        service = create_service(settings, args.top_k)
    except (ConfigError, KnowledgeBaseError, ValueError) as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        return 1

    show_prompt = not args.hide_prompt
    if args.question:
        for question in args.question:
            ask(service, settings, question, show_prompt)
        return 0

    run_interactive(service, settings, show_prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
