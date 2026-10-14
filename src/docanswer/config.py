"""Runtime configuration from environment variables (12-factor). No secrets are logged."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = Path("data/docs")
    index_dir: Path = Path("data/index")
    # Providers are opt-in: the default is a fully offline, deterministic stack.
    llm_provider: str = "extractive"  # extractive | anthropic | openai
    llm_model: str = ""
    embedder: str = "tfidf"  # tfidf | openai | none
    top_k: int = 5
    refusal_threshold: float = 0.47  # calibrated on eval/dev_questions.jsonl
    chunk_max_words: int = 90
    chunk_overlap_sentences: int = 1
    ingest_token: str = ""

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            docs_dir=Path(_env("DOCS_DIR", "data/docs")),
            index_dir=Path(_env("INDEX_DIR", "data/index")),
            llm_provider=_env("LLM_PROVIDER", "extractive").lower(),
            llm_model=_env("LLM_MODEL", ""),
            embedder=_env("EMBEDDER", "tfidf").lower(),
            top_k=int(_env("TOP_K", "5")),
            refusal_threshold=float(_env("REFUSAL_THRESHOLD", "0.47")),
            chunk_max_words=int(_env("CHUNK_MAX_WORDS", "90")),
            chunk_overlap_sentences=int(_env("CHUNK_OVERLAP_SENTENCES", "1")),
            ingest_token=os.environ.get("INGEST_TOKEN", ""),
        )
