"""Tests for document loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.knowledge_base import Document, KnowledgeBaseError, load_documents
from tests.conftest import valid_record

REQUIRED_TITLES = {
    "DOC001": "Donor Retention",
    "DOC002": "Donor Acquisition",
    "DOC003": "Data Privacy",
    "DOC004": "Campaign Performance",
    "DOC005": "Donor Segmentation",
}


def test_project_knowledge_base_contains_required_documents(
    project_documents: list[Document],
) -> None:
    titles = {document.id: document.title for document in project_documents}
    for doc_id, title in REQUIRED_TITLES.items():
        assert titles[doc_id] == title
    assert len(project_documents) >= 5


def test_every_project_document_is_marked_synthetic(project_documents: list[Document]) -> None:
    assert all(d.source_type == "synthetic_training_document" for d in project_documents)


def test_documents_are_sorted_by_id(project_documents: list[Document]) -> None:
    ids = [document.id for document in project_documents]
    assert ids == sorted(ids)


def test_non_json_files_are_ignored(write_json, tmp_path: Path) -> None:
    write_json("a.json", valid_record("A001"))
    (tmp_path / "README.md").write_text("# not a document", encoding="utf-8")
    assert [d.id for d in load_documents(tmp_path)] == ["A001"]


def test_json_file_may_contain_a_list_of_documents(write_json) -> None:
    directory = write_json("many.json", [valid_record("B002"), valid_record("B001")])
    assert [d.id for d in load_documents(directory)] == ["B001", "B002"]


def test_fields_are_stripped(write_json) -> None:
    directory = write_json("a.json", valid_record(title="  Padded Title  "))
    assert load_documents(directory)[0].title == "Padded Title"


def test_missing_required_field_raises(write_json) -> None:
    record = valid_record()
    del record["content"]
    directory = write_json("bad.json", record)
    with pytest.raises(KnowledgeBaseError, match="missing required fields"):
        load_documents(directory)


def test_blank_required_field_raises(write_json) -> None:
    directory = write_json("bad.json", valid_record(title="   "))
    with pytest.raises(KnowledgeBaseError, match="'title'"):
        load_documents(directory)


def test_invalid_json_raises(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="invalid JSON"):
        load_documents(tmp_path)


def test_duplicate_ids_raise(write_json) -> None:
    write_json("one.json", valid_record("D001"))
    directory = write_json("two.json", valid_record("D001"))
    with pytest.raises(KnowledgeBaseError, match="Duplicate document IDs"):
        load_documents(directory)


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeBaseError, match="not found"):
        load_documents(tmp_path / "does-not-exist")


def test_empty_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeBaseError, match="No documents"):
        load_documents(tmp_path)
