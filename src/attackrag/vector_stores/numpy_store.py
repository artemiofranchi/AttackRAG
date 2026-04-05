from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from attackrag.chunking import Chunk
from attackrag.vector_stores.meta import write_meta
from attackrag.vector_stores.types import RetrievedChunk


@dataclass
class NumpyVectorStore:
    vectors: np.ndarray
    chunks: list[Chunk]
    embedding_model: str | None = None

    def save(self, directory: Path) -> None:
        persist_numpy(directory, self.chunks, self.vectors, self.embedding_model)

    @classmethod
    def load(cls, directory: Path, meta: dict) -> NumpyVectorStore:
        vectors = np.load(directory / "vectors.npy")
        raw = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in raw]
        return cls(
            vectors=vectors,
            chunks=chunks,
            embedding_model=meta.get("embedding_model"),
        )

    def search(self, query_embedding: np.ndarray, k: int) -> list[RetrievedChunk]:
        if self.vectors.size == 0:
            return []
        q = query_embedding.astype(np.float32, copy=False).reshape(1, -1)
        sims = (self.vectors @ q.T).reshape(-1)
        k = min(k, sims.shape[0])
        top_idx = np.argpartition(-sims, kth=k - 1)[-k:]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        out: list[RetrievedChunk] = []
        for i in top_idx:
            ch = self.chunks[int(i)]
            out.append(
                RetrievedChunk(
                    chunk_id=ch.chunk_id,
                    doc_id=ch.doc_id,
                    text=ch.text,
                    score=float(sims[int(i)]),
                )
            )
        return out


def persist_numpy(
    directory: Path,
    chunks: list[Chunk],
    vectors: np.ndarray,
    embedding_model: str | None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "vectors.npy", vectors)
    payload = [asdict(c) for c in chunks]
    (directory / "chunks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_meta(
        directory,
        {
            "backend": "numpy",
            "embedding_model": embedding_model,
            "vector_size": int(vectors.shape[1]),
        },
    )
