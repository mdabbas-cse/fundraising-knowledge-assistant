"""Application settings.

All tunable values live here so they can be changed without touching code.
Values come from environment variables (optionally loaded from a `.env` file by
`main.py`) and fall back to sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


class ConfigError(ValueError):
    """Raised when a setting has an invalid value."""


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for the assistant."""

    documents_dir: Path = DEFAULT_DOCUMENTS_DIR
    # How many documents the retriever returns at most.
    top_k: int = 3
    # Cosine similarity below this value is treated as "not relevant".
    # Tuned on the synthetic corpus: the best match for each sample question
    # scores ~0.24-0.32, partial single-word matches score ~0.02-0.06 and
    # fully off-topic questions score 0.0. See README "Similarity threshold".
    min_similarity: float = 0.10
    # Soft upper bound on chunk length (words). Long documents are split.
    chunk_max_words: int = 80
    # Number of trailing sentences repeated at the start of the next chunk.
    chunk_overlap_sentences: int = 1

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ConfigError(f"top_k must be >= 1, got {self.top_k}")
        if not 0.0 <= self.min_similarity <= 1.0:
            raise ConfigError(f"min_similarity must be between 0 and 1, got {self.min_similarity}")
        if self.chunk_max_words < 1:
            raise ConfigError(f"chunk_max_words must be >= 1, got {self.chunk_max_words}")
        if self.chunk_overlap_sentences < 0:
            raise ConfigError(
                f"chunk_overlap_sentences must be >= 0, got {self.chunk_overlap_sentences}"
            )


def _read_env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _parse_int(name: str, default: int) -> int:
    raw = _read_env(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _parse_float(name: str, default: float) -> float:
    raw = _read_env(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def load_settings() -> Settings:
    """Build settings from environment variables (prefix `FKA_`)."""
    defaults = Settings()
    documents_dir = Path(_read_env("FKA_DOCUMENTS_DIR", str(defaults.documents_dir)))
    if not documents_dir.is_absolute():
        # Resolve relative paths against the project root, not the current
        # working directory, so `python main.py` works from any folder.
        documents_dir = PROJECT_ROOT / documents_dir

    return Settings(
        documents_dir=documents_dir,
        top_k=_parse_int("FKA_TOP_K", defaults.top_k),
        min_similarity=_parse_float("FKA_MIN_SIMILARITY", defaults.min_similarity),
        chunk_max_words=_parse_int("FKA_CHUNK_MAX_WORDS", defaults.chunk_max_words),
        chunk_overlap_sentences=_parse_int(
            "FKA_CHUNK_OVERLAP_SENTENCES", defaults.chunk_overlap_sentences
        ),
    )
