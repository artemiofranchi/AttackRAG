"""Topic-Consistent Re-ranking (TCR) — стадия 2; разделяемая с RAGFort-профилем."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from attackrag.chunking import Chunk
from attackrag.embeddings import EmbeddingModel
from attackrag.vector_stores.types import RetrievedChunk


@dataclass
class TCRMeta:
    k_topics: int
    centroids: list[list[float]]
    topic_by_chunk_id: dict[str, int]
    embedding_model: str
    version: int = 1


class TopicConsistentReranker:
    def __init__(
        self,
        embedder: EmbeddingModel,
        *,
        k_topics: int = 12,
        alpha: float = 0.6,
        beta: float = 0.3,
        gamma: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self._embedder = embedder
        self.k_topics = k_topics
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._rs = random_state
        self._kmeans: KMeans | None = None
        self._centroids: np.ndarray | None = None
        self._topic_by_id: dict[str, int] = {}
        self._chunk_emb_by_id: dict[str, np.ndarray] = {}
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, chunks: list[Chunk], *, seed: int | None = None) -> None:
        if not chunks:
            return
        rs = int(seed) if seed is not None else self._rs
        texts = [c.text for c in chunks]
        X = self._embedder.encode(texts)
        k = min(self.k_topics, max(1, len(chunks)))
        km = KMeans(n_clusters=k, random_state=rs, max_iter=300, n_init="auto")
        lab = km.fit_predict(X)
        self._kmeans = km
        self._centroids = km.cluster_centers_.astype(np.float32, copy=False)
        self._topic_by_id = {c.chunk_id: int(lab[i]) for i, c in enumerate(chunks)}
        self._chunk_emb_by_id = {c.chunk_id: X[i] for i, c in enumerate(chunks)}
        self._fitted = True

    def save(self, index_dir: Path) -> None:
        if not self._fitted or self._centroids is None:
            return
        p = index_dir / "tcr_meta.json"
        m = TCRMeta(
            k_topics=self._centroids.shape[0],
            centroids=[row.tolist() for row in self._centroids],
            topic_by_chunk_id={**self._topic_by_id},
            embedding_model=self._embedder.model_name,
        )
        p.write_text(json.dumps(asdict(m), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path, embedder: EmbeddingModel) -> TopicConsistentReranker:
        p = index_dir / "tcr_meta.json"
        m = TCRMeta(**json.loads(p.read_text(encoding="utf-8")))
        t = cls(embedder, k_topics=m.k_topics)
        t._centroids = np.array(m.centroids, dtype=np.float32)
        t._topic_by_id = {**m.topic_by_chunk_id}
        t._chunk_emb_by_id = {}
        t._fitted = True
        t._kmeans = None
        return t

    def _q_topic(self, query: str) -> int:
        if self._centroids is None or self._centroids.size == 0:
            return 0
        qe = self._embedder.encode([query])[0]
        d = np.linalg.norm(self._centroids - qe[None, :], axis=1)
        return int(d.argmin())

    def anom_score(self, chunk_id: str) -> float:
        if not self._fitted or self._centroids is None or chunk_id not in self._topic_by_id:
            return 0.0
        t = self._topic_by_id[chunk_id]
        c = self._centroids[t]
        emb = self._chunk_emb_by_id.get(chunk_id)
        if emb is None:
            return 0.0
        d = float(np.linalg.norm(emb - c) / (np.linalg.norm(emb) + 1e-9))
        return float(np.clip(d, 0.0, 1.0))

    def context_anom(self, hits: list[RetrievedChunk]) -> float:
        if not hits:
            return 0.0
        return float(np.mean([self.anom_score(h.chunk_id) for h in hits]))

    def rerank(self, query: str, hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not self._fitted or not hits or self._centroids is None:
            return hits
        qt = self._q_topic(query)
        scored: list[tuple[float, RetrievedChunk]] = []
        for h in hits:
            t_h = self._topic_by_id.get(h.chunk_id, 0)
            t_sim = 1.0 if t_h == qt else 0.2
            an = self.anom_score(h.chunk_id)
            s = self.alpha * float(h.score) + self.beta * t_sim + self.gamma * (1.0 - an)
            scored.append((s, h))
        scored.sort(key=lambda x: -x[0])
        return [h for _s, h in scored]
