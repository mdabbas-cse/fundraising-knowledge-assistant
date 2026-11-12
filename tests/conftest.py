"""Shared pytest fixtures."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.config import Settings
from app.document_processor import process_documents
from app.knowledge_base import Document, load_documents
from app.retriever import TfidfRetriever, build_retriever

# The five practical questions from the project brief and the document each
# one is expected to retrieve first.
REQUIRED_QUESTIONS: dict[str, str] = {
    "How can we improve donor retention?": "DOC001",
    "How can we measure campaign performance?": "DOC004",
    "What is donor segmentation?": "DOC005",
    "How can we protect personal information?": "DOC003",
    "What are effective donor acquisition strategies?": "DOC002",
}

UNRELATED_QUESTIONS: list[str] = [
    "What is the best recipe for chocolate cake?",
    "Who won the football world cup?",
    "What is the capital of France?",
]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test with default settings, ignoring the developer's .env / shell."""
    for name in list(os.environ):
        if name.startswith("FKA_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr("main.load_dotenv", lambda *args, **kwargs: False)


def make_document(doc_id: str, title: str, content: str) -> Document:
    return Document(
        id=doc_id,
        title=title,
        category="test",
        content=content,
        created_at="2026-01-01",
        source_type="synthetic_training_document",
    )


@pytest.fixture
def sample_documents() -> list[Document]:
    """A tiny, fully controlled synthetic corpus."""
    return [
        make_document(
            "T001",
            "Donor Retention",
            "Thank donors quickly. Send impact updates to keep donors giving every year.",
        ),
        make_document(
            "T002",
            "Data Privacy",
            "Encrypt personal information. Limit access to donor records.",
        ),
        make_document(
            "T003",
            "Volunteer Events",
            "Volunteers organise charity runs and bake sales in local communities.",
        ),
    ]


@pytest.fixture
def small_retriever(sample_documents: list[Document]) -> TfidfRetriever:
    chunks = process_documents(sample_documents, max_words=80, overlap_sentences=0)
    return TfidfRetriever(chunks, min_similarity=0.10)


@pytest.fixture(scope="session")
def project_settings() -> Settings:
    """Default settings pointing at the real synthetic knowledge base."""
    return Settings()


@pytest.fixture(scope="session")
def project_documents(project_settings: Settings) -> list[Document]:
    return load_documents(project_settings.documents_dir)


@pytest.fixture(scope="session")
def project_retriever(project_settings: Settings) -> TfidfRetriever:
    return build_retriever(project_settings)


@pytest.fixture
def write_json(tmp_path: Path):
    """Helper that writes a JSON payload to tmp_path/<name> and returns the dir."""

    def _write(name: str, payload: object) -> Path:
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
        return tmp_path

    return _write


def valid_record(doc_id: str = "X001", **overrides: str) -> dict[str, str]:
    record = {
        "id": doc_id,
        "title": "Test Title",
        "category": "test",
        "content": "Some synthetic content.",
        "created_at": "2026-01-01",
        "source_type": "synthetic_training_document",
    }
    record.update(overrides)
    return record
