"""Плюгинные векторные хранилища (ближе к продакшену для темы атак на retrieval/индекс)."""

from attackrag.vector_stores.loader import (
    BACKENDS,
    BackendName,
    build_vector_store,
    load_vector_store,
)
from attackrag.vector_stores.types import RetrievedChunk

__all__ = [
    "BACKENDS",
    "BackendName",
    "RetrievedChunk",
    "build_vector_store",
    "load_vector_store",
]
