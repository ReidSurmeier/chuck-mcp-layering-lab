"""pytest fixtures for the v23-MCP test rings.

Ring 1: direct python (`tests/v23/direct/` and `tests/v23/scaffold/`).
Ring 2: transport (`tests/v23/transport/`).
Ring 3: mock-Opus conversation (`tests/v23/conversation/`).
Ring 4: per-stage (`tests/v23/stages/`).
Ring 5: corpus regression (`tests/v23/corpus/`).
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def corpus_dir(repo_root: Path) -> Path:
    return repo_root / "corpus"
