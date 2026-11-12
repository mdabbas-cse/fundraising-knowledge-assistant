"""Smoke tests for the command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest

import main


def test_one_off_question_prints_documents_prompt_and_sources(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main.main(["-q", "What is donor segmentation?"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Relevant Documents:" in output
    assert "ID: DOC005" in output
    assert "Similarity:" in output
    assert "Generated LLM Prompt:" in output
    assert "- DOC005: Donor Segmentation" in output


def test_unrelated_question_reports_no_documents(capsys: pytest.CaptureFixture[str]) -> None:
    main.main(["-q", "What is the capital of France?", "--hide-prompt"])
    output = capsys.readouterr().out
    assert "None. Every similarity score was below the threshold" in output
    assert "Generated LLM Prompt:" not in output


def test_interactive_mode_handles_several_questions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = iter(
        ["How can we improve donor retention?", "", "What is donor segmentation?", "exit"]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert main.main(["--hide-prompt"]) == 0
    output = capsys.readouterr().out
    assert "ID: DOC001" in output
    assert "Invalid question" in output
    assert "ID: DOC005" in output
    assert "Goodbye." in output


def test_missing_knowledge_base_returns_error_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FKA_DOCUMENTS_DIR", str(tmp_path / "missing"))
    assert main.main(["-q", "donor retention"]) == 1
    assert "Startup error" in capsys.readouterr().err
