from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_corpus_dir() -> Path:
    return repo_root() / "data" / "corpus"


def default_golden_qa_path() -> Path:
    return repo_root() / "data" / "golden_qa.json"


def default_index_dir() -> Path:
    return repo_root() / "data" / "index"
