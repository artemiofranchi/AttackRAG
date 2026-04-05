from __future__ import annotations

from pathlib import Path
from typing import Literal

from attackrag.chunking import Chunk
from attackrag.embeddings import EmbeddingModel
from attackrag.vector_stores.chroma_store import persist_chroma
from attackrag.vector_stores.faiss_store import persist_faiss
from attackrag.vector_stores.meta import read_meta
from attackrag.vector_stores.numpy_store import NumpyVectorStore, persist_numpy
from attackrag.vector_stores.protocol import VectorStore
from attackrag.vector_stores.qdrant_store import persist_qdrant

BackendName = Literal["numpy", "faiss", "chroma", "qdrant"]

BACKENDS: tuple[BackendName, ...] = ("numpy", "faiss", "chroma", "qdrant")


def _infer_backend(directory: Path, meta: dict) -> BackendName:
    b = meta.get("backend")
    if b in BACKENDS:
        return b  # type: ignore[return-value]
    if (directory / "vectors.npy").is_file():
        return "numpy"
    if (directory / "faiss.index").is_file():
        return "faiss"
    if (directory / "chroma_db").is_dir():
        return "chroma"
    if (directory / "qdrant_storage").is_dir():
        return "qdrant"
    raise FileNotFoundError(
        f"Не удалось определить тип индекса в {directory}: нет meta.backend и известных артефактов"
    )


def load_vector_store(directory: Path | str) -> VectorStore:
    d = Path(directory)
    meta = read_meta(d)
    backend = _infer_backend(d, meta)
    if backend == "numpy":
        return NumpyVectorStore.load(d, meta)
    if backend == "faiss":
        from attackrag.vector_stores.faiss_store import FaissVectorStore

        return FaissVectorStore.load(d, meta)
    if backend == "chroma":
        from attackrag.vector_stores.chroma_store import ChromaVectorStore

        return ChromaVectorStore.load(d, meta)
    if backend == "qdrant":
        from attackrag.vector_stores.qdrant_store import QdrantVectorStore

        return QdrantVectorStore.load(d, meta)
    raise ValueError(f"Неизвестный backend: {backend}")


def build_vector_store(
    out_dir: Path | str,
    chunks: list[Chunk],
    embedder: EmbeddingModel,
    backend: BackendName = "numpy",
) -> None:
    texts = [c.text for c in chunks]
    vectors = embedder.encode(texts)
    d = Path(out_dir)
    if backend == "numpy":
        persist_numpy(d, chunks, vectors, embedder.model_name)
    elif backend == "faiss":
        persist_faiss(d, chunks, vectors, embedder.model_name)
    elif backend == "chroma":
        persist_chroma(d, chunks, vectors, embedder.model_name)
    elif backend == "qdrant":
        persist_qdrant(d, chunks, vectors, embedder.model_name)
    else:
        raise ValueError(f"Неизвестный backend: {backend}")
