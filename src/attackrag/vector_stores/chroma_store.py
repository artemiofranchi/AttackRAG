from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from attackrag.chunking import Chunk
from attackrag.vector_stores.meta import write_meta
from attackrag.vector_stores.types import RetrievedChunk


def _require_chroma():
    try:
        import chromadb  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Chroma: установите зависимость: pip install attack-rag[chroma]"
        ) from e


COLLECTION = "attackrag"


@dataclass
class ChromaVectorStore:
    _client: object
    _collection: object
    chunks: list[Chunk]
    embedding_model: str | None = None

    @classmethod
    def load(cls, directory: Path, meta: dict) -> ChromaVectorStore:
        _require_chroma()
        import chromadb

        raw = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in raw]
        client = chromadb.PersistentClient(path=str(directory / "chroma_db"))
        collection = client.get_collection(COLLECTION)
        return cls(
            _client=client,
            _collection=collection,
            chunks=chunks,
            embedding_model=meta.get("embedding_model"),
        )

    def search(self, query_embedding: np.ndarray, k: int) -> list[RetrievedChunk]:
        n = self._collection.count()
        if n == 0:
            return []
        q = query_embedding.astype(np.float32, copy=False).tolist()
        res = self._collection.query(
            query_embeddings=[q],
            n_results=min(k, n),
            include=["distances", "documents", "metadatas"],
        )
        ids_order = res["ids"][0]
        dists = res["distances"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        chunk_by_id = {c.chunk_id: c for c in self.chunks}
        out: list[RetrievedChunk] = []
        for cid, dist, text, meta in zip(ids_order, dists, docs, metas):
            ch = chunk_by_id.get(str(cid))
            if ch is None and meta:
                ch = Chunk(
                    chunk_id=str(meta.get("chunk_id", cid)),
                    doc_id=str(meta.get("doc_id", "")),
                    text=text or "",
                )
            if ch is None:
                continue
            score = float(1.0 - dist) if dist is not None else 0.0
            out.append(
                RetrievedChunk(
                    chunk_id=ch.chunk_id,
                    doc_id=ch.doc_id,
                    text=ch.text,
                    score=score,
                )
            )
        return out


def persist_chroma(
    directory: Path,
    chunks: list[Chunk],
    vectors: np.ndarray,
    embedding_model: str | None,
) -> None:
    _require_chroma()
    import chromadb

    directory.mkdir(parents=True, exist_ok=True)
    payload = [asdict(c) for c in chunks]
    (directory / "chunks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    chroma_path = directory / "chroma_db"
    if chroma_path.exists():
        import shutil

        shutil.rmtree(chroma_path)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    ids = [c.chunk_id for c in chunks]
    metadatas = [{"chunk_id": c.chunk_id, "doc_id": c.doc_id} for c in chunks]
    texts = [c.text for c in chunks]
    emb = vectors.astype(np.float32, copy=False).tolist()
    collection.add(ids=ids, embeddings=emb, documents=texts, metadatas=metadatas)
    write_meta(
        directory,
        {
            "backend": "chroma",
            "embedding_model": embedding_model,
            "vector_size": int(vectors.shape[1]),
        },
    )
