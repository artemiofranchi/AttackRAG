from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RawDocument:
    doc_id: str
    path: Path
    text: str


def load_markdown_corpus(corpus_dir: Path) -> list[RawDocument]:
    corpus_dir = corpus_dir.resolve()
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Каталог корпуса не найден: {corpus_dir}")
    docs: list[RawDocument] = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append(RawDocument(doc_id=path.stem, path=path, text=text))
    if not docs:
        raise ValueError(f"В {corpus_dir} нет ни одного .md файла с текстом")
    return docs
