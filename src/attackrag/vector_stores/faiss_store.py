from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from attackrag.chunking import Chunk
from attackrag.vector_stores.meta import write_meta
from attackrag.vector_stores.types import RetrievedChunk


def _require_faiss():
    try:
        import faiss  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "FAISS: установите зависимость: pip install attack-rag[faiss]  (или faiss-cpu)"
        ) from e
    return __import__("faiss")


@dataclass
class FaissVectorStore:
    index: object  # faiss.IndexFlatIP
    chunks: list[Chunk]
    embedding_model: str | None = None

    @classmethod
    def load(cls, directory: Path, meta: dict) -> FaissVectorStore:
        faiss = _require_faiss()
        idx = faiss.read_index(str(directory / "faiss.index"))
        raw = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in raw]
        return cls(index=idx, chunks=chunks, embedding_model=meta.get("embedding_model"))

    def search(self, query_embedding: np.ndarray, k: int) -> list[RetrievedChunk]:
        faiss = _require_faiss()
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        q = query_embedding.astype(np.float32, copy=False).reshape(1, -1)
        sims, idxs = self.index.search(q, k)
        sims_row = sims[0]
        idxs_row = idxs[0]
        out: list[RetrievedChunk] = []
        for rank in range(k):
            i = int(idxs_row[rank])
            if i < 0:
                continue
            ch = self.chunks[i]
            out.append(
                RetrievedChunk(
                    chunk_id=ch.chunk_id,
                    doc_id=ch.doc_id,
                    text=ch.text,
                    score=float(sims_row[rank]),
                )
            )
        return out


def persist_faiss(
    directory: Path,
    chunks: list[Chunk],
    vectors: np.ndarray,
    embedding_model: str | None,
) -> None:
    faiss = _require_faiss()
    directory.mkdir(parents=True, exist_ok=True)
    dim = int(vectors.shape[1])
    mat = vectors.astype(np.float32, copy=False)
    index = faiss.IndexFlatIP(dim)
    index.add(mat)
    faiss.write_index(index, str(directory / "faiss.index"))
    payload = [asdict(c) for c in chunks]
    (directory / "chunks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_meta(
        directory,
        {
            "backend": "faiss",
            "embedding_model": embedding_model,
            "vector_size": dim,
        },
    )
