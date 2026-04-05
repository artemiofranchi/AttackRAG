from __future__ import annotations

import tempfile
from pathlib import Path

from attackrag.attacks.corpus_poison import copy_corpus_with_poison, write_poison_file
from attackrag.chunking import chunk_documents
from attackrag.documents import load_markdown_corpus
from attackrag.embeddings import EmbeddingModel
from attackrag.vector_stores.loader import BACKENDS, build_vector_store


def build_poisoned_index(
    base_corpus: Path,
    out_index: Path,
    *,
    trigger: str,
    secret: str,
    backend: str,
    embedding_model: str,
    max_chars: int,
    overlap: int,
    merged_corpus_dir: Path | None = None,
) -> Path:
    """Копирует base_corpus + poison .md и собирает новый индекс в out_index (отдельный от baseline)."""
    if backend not in BACKENDS:
        raise ValueError(f"backend должен быть одним из {BACKENDS}")
    merged = merged_corpus_dir or (out_index.parent / f"corpus_merged_{out_index.name}")
    with tempfile.TemporaryDirectory() as td:
        tmp_poison = Path(td) / "_poison_backdoor.md"
        write_poison_file(tmp_poison, trigger=trigger, secret_value=secret)
        copy_corpus_with_poison(base_corpus, tmp_poison, merged)
    docs = load_markdown_corpus(merged)
    chunks = chunk_documents(docs, max_chars=max_chars, overlap=overlap)
    embedder = EmbeddingModel(embedding_model)
    build_vector_store(out_index, chunks, embedder, backend=backend)  # type: ignore[arg-type]
    return merged
