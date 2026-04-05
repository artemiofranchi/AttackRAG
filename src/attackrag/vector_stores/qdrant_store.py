from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from attackrag.chunking import Chunk
from attackrag.vector_stores.meta import write_meta
from attackrag.vector_stores.types import RetrievedChunk


def _require_qdrant():
    try:
        import qdrant_client  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Qdrant: установите зависимость: pip install attack-rag[qdrant]"
        ) from e


COLLECTION = "attackrag"


@dataclass
class QdrantVectorStore:
    _client: object
    chunks: list[Chunk]
    embedding_model: str | None = None

    @classmethod
    def load(cls, directory: Path, meta: dict) -> QdrantVectorStore:
        _require_qdrant()
        from qdrant_client import QdrantClient

        raw = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in raw]
        client = QdrantClient(path=str(directory / "qdrant_storage"))
        return cls(_client=client, chunks=chunks, embedding_model=meta.get("embedding_model"))

    def search(self, query_embedding: np.ndarray, k: int) -> list[RetrievedChunk]:
        n = len(self.chunks)
        if n == 0:
            return []
        q = query_embedding.astype(np.float32, copy=False).tolist()
        res = self._client.query_points(
            collection_name=COLLECTION,
            query=q,
            limit=min(k, n),
            with_payload=True,
        )
        out: list[RetrievedChunk] = []
        for hit in res.points:
            pl = hit.payload or {}
            out.append(
                RetrievedChunk(
                    chunk_id=str(pl.get("chunk_id", "")),
                    doc_id=str(pl.get("doc_id", "")),
                    text=str(pl.get("text", "")),
                    score=float(hit.score or 0.0),
                )
            )
        return out


def persist_qdrant(
    directory: Path,
    chunks: list[Chunk],
    vectors: np.ndarray,
    embedding_model: str | None,
) -> None:
    _require_qdrant()
    import shutil

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    directory.mkdir(parents=True, exist_ok=True)
    payload = [asdict(c) for c in chunks]
    (directory / "chunks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    qpath = directory / "qdrant_storage"
    if qpath.exists():
        shutil.rmtree(qpath)
    client = QdrantClient(path=str(qpath))
    dim = int(vectors.shape[1])
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].astype(np.float32, copy=False).tolist(),
            payload={
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
            },
        )
        for i, c in enumerate(chunks)
    ]
    client.upsert(collection_name=COLLECTION, points=points, wait=True)
    write_meta(
        directory,
        {
            "backend": "qdrant",
            "embedding_model": embedding_model,
            "vector_size": dim,
        },
    )
