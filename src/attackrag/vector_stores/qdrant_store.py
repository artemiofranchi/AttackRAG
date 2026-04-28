from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
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


def _open_local_qdrant_path(qdrant_dir: Path) -> Path:
    """
    Локальный ``QdrantClient(path=...)`` держит эксклюзивную блокировку каталога хранилища;
    второй процесс получает ``AlreadyLocked``. Если нужно параллельно открыть один и тот же
    индекс (или обойти «залипшую» блокировку после сбоя), задайте окружение::

        ATTACKRAG_QDRANT_ISOLATE_COPY=1

    Тогда перед открытием делается копия ``qdrant_storage`` во временный каталог (дольше старт,
    больше места на диске). Для параллельных прогонов надёжнее отдельный Qdrant Server (Docker).
    """
    v = os.environ.get("ATTACKRAG_QDRANT_ISOLATE_COPY", "").strip().lower()
    if v not in ("1", "true", "yes"):
        return qdrant_dir
    dst = Path(tempfile.gettempdir()) / f"attackrag_qdrant_{uuid.uuid4().hex}"
    shutil.copytree(qdrant_dir, dst)
    return dst


def _qdrant_client_from_env_or_path(qpath: Path):
    """Открывает QdrantClient: server (URL из env) или embedded (file lock).

    Если задан ``QDRANT_URL`` (например, выставлен в Streamlit-UI или экспортирован
    в shell), идём в server-режим — он поддерживает параллельных клиентов и
    устраняет ``RuntimeError: Storage folder is already accessed by another instance``,
    которая ловила embedded-режим при оркестрации (несколько subprocess подряд).
    Иначе используется embedded — но с понятной диагностикой при коллизии lock.
    """
    from qdrant_client import QdrantClient

    url = (os.environ.get("QDRANT_URL") or "").strip()
    if url:
        api_key = os.environ.get("QDRANT_API_KEY") or None
        return QdrantClient(url=url, api_key=api_key)
    try:
        return QdrantClient(path=str(qpath))
    except RuntimeError as e:
        # Превращаем raw-сообщение portalocker в actionable hint для пользователя:
        # это тот самый кейс, когда параллельно работает UI или зомби-процесс.
        msg = (
            f"Embedded Qdrant занят другим процессом ({qpath}).\n"
            "Варианты:\n"
            "  1) Закройте UI и/или другие процессы, держащие lock:\n"
            f"       lsof | grep {qpath.name}\n"
            "  2) Поднимите Qdrant в Docker и задайте переменную окружения:\n"
            "       docker compose up -d qdrant\n"
            "       export QDRANT_URL=http://localhost:6333\n"
            "  3) Включите изолированную копию (медленнее, но позволяет параллельный read-only):\n"
            "       export ATTACKRAG_QDRANT_ISOLATE_COPY=1"
        )
        raise RuntimeError(msg) from e


@dataclass
class QdrantVectorStore:
    _client: object
    chunks: list[Chunk]
    embedding_model: str | None = None

    @classmethod
    def load(cls, directory: Path, meta: dict) -> QdrantVectorStore:
        _require_qdrant()

        raw = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in raw]
        qpath = _open_local_qdrant_path(directory / "qdrant_storage")
        client = _qdrant_client_from_env_or_path(qpath)
        return cls(_client=client, chunks=chunks, embedding_model=meta.get("embedding_model"))

    def close(self) -> None:
        """Явно закрываем embedded-Qdrant клиент — освобождает file-lock.

        Без этого `QdrantLocal.__del__` срабатывает только в GC, а в Python 3.13
        при быстром выходе подпроцесса GC может не успеть выполниться, и lock
        утекает — следующий subprocess `attack-rag-run-attacks` получает
        `AlreadyLocked: Resource temporarily unavailable` (точно такой же кейс,
        что в логах прогона 21:01:55).
        """
        client = getattr(self, "_client", None)
        if client is None:
            return
        try:
            client.close()
        except Exception:  # noqa: BLE001 — best-effort, lock уже не освободить иначе
            pass

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
