"""Knowledge base: load raw documents from disk and validate them.

Only JSON is supported in Phase 1. Support for other formats is added by writing
one loader function and registering it in `LOADERS`, for example:

    def load_pdf_file(path: Path) -> list[Document]:
        text = extract_text_with_pypdf(path)
        return [Document(id=..., title=path.stem, content=text, ...)]

    LOADERS[".pdf"] = load_pdf_file
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "title",
    "category",
    "content",
    "created_at",
    "source_type",
)


class KnowledgeBaseError(Exception):
    """Raised when documents cannot be loaded or fail validation."""


@dataclass(frozen=True)
class Document:
    """One knowledge document exactly as stored in the knowledge base."""

    id: str
    title: str
    category: str
    content: str
    created_at: str
    source_type: str


def document_from_dict(raw: dict[str, Any], source: str = "<memory>") -> Document:
    """Validate a raw dict and convert it to a `Document`.

    Extra keys (such as `disclaimer`) are allowed and ignored.
    """
    if not isinstance(raw, dict):
        raise KnowledgeBaseError(f"{source}: expected a JSON object, got {type(raw).__name__}")

    missing = [field for field in REQUIRED_FIELDS if field not in raw]
    if missing:
        raise KnowledgeBaseError(f"{source}: missing required fields {missing}")

    values: dict[str, str] = {}
    for field in REQUIRED_FIELDS:
        value = raw[field]
        if not isinstance(value, str) or not value.strip():
            raise KnowledgeBaseError(f"{source}: field '{field}' must be a non-empty string")
        values[field] = value.strip()

    return Document(**values)


def load_json_file(path: Path) -> list[Document]:
    """Load one JSON file containing a document object or a list of them."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise KnowledgeBaseError(f"{path.name}: invalid JSON ({exc})") from exc
    except OSError as exc:
        raise KnowledgeBaseError(f"{path.name}: cannot read file ({exc})") from exc

    records = payload if isinstance(payload, list) else [payload]
    return [document_from_dict(record, source=path.name) for record in records]


# File extension -> loader function. Add ".pdf" / ".docx" loaders in Phase 2.
LOADERS: dict[str, Callable[[Path], list[Document]]] = {
    ".json": load_json_file,
}


def load_documents(documents_dir: Path) -> list[Document]:
    """Load every supported file in `documents_dir`.

    Files with unsupported extensions (for example README.md) are skipped.
    Documents are returned sorted by ID so results are deterministic.
    """
    if not documents_dir.is_dir():
        raise KnowledgeBaseError(f"Documents directory not found: {documents_dir}")

    documents: list[Document] = []
    for path in sorted(documents_dir.iterdir()):
        loader = LOADERS.get(path.suffix.lower())
        if loader is not None and path.is_file():
            documents.extend(loader(path))

    if not documents:
        raise KnowledgeBaseError(f"No documents found in {documents_dir}")

    _ensure_unique_ids(documents)
    return sorted(documents, key=lambda document: document.id)


def _ensure_unique_ids(documents: list[Document]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for document in documents:
        if document.id in seen:
            duplicates.add(document.id)
        seen.add(document.id)
    if duplicates:
        raise KnowledgeBaseError(f"Duplicate document IDs: {sorted(duplicates)}")
