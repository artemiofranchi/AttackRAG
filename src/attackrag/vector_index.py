from __future__ import annotations

"""
Обратная совместимость: in-memory NumPy-индекс и алиасы.

Персистентные индексы и выбор backend: `attackrag.vector_stores`.
"""

from attackrag.chunking import Chunk
from attackrag.embeddings import EmbeddingModel
from attackrag.vector_stores.numpy_store import NumpyVectorStore
from attackrag.vector_stores.types import RetrievedChunk

# Старый код мог ссылаться на PersistentIndex / retrieve_top_k.
PersistentIndex = NumpyVectorStore


def build_index(
    chunks: list[Chunk],
    embedder: EmbeddingModel,
) -> NumpyVectorStore:
    texts = [c.text for c in chunks]
    vectors = embedder.encode(texts)
    return NumpyVectorStore(
        vectors=vectors,
        chunks=chunks,
        embedding_model=embedder.model_name,
    )


def retrieve_top_k(
    index: NumpyVectorStore,
    query_embedding,
    k: int,
) -> list[RetrievedChunk]:
    return index.search(query_embedding, k)


__all__ = [
    "PersistentIndex",
    "RetrievedChunk",
    "NumpyVectorStore",
    "build_index",
    "retrieve_top_k",
]
