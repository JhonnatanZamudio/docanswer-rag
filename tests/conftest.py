from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from docanswer.config import Settings
from docanswer.ingest import Index, build_index
from docanswer.pipeline import DocAnswer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    # Offline, deterministic settings regardless of the developer's environment variables.
    return replace(
        Settings(),
        docs_dir=ROOT / "data" / "docs",
        index_dir=tmp_path_factory.mktemp("index"),
        llm_provider="extractive",
        embedder="tfidf",
    )


@pytest.fixture(scope="session")
def index(settings: Settings) -> Index:
    idx = build_index(settings)
    idx.save(settings.index_dir)
    return idx


@pytest.fixture(scope="session")
def engine(settings: Settings, index: Index) -> DocAnswer:
    return DocAnswer.from_settings(settings)
