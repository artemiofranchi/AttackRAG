from __future__ import annotations

from typing import Protocol

import numpy as np

from attackrag.vector_stores.types import RetrievedChunk


class VectorStore(Protocol):
    """Единый интерфейс retrieval для RAG и экспериментов по атакам на индекс."""

    embedding_model: str | None

    def search(self, query_embedding: np.ndarray, k: int) -> list[RetrievedChunk]: ...
