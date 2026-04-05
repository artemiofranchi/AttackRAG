from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attackrag.embeddings import EmbeddingModel


@dataclass
class ControlNetProxyDefense:
    """ControlNET-style proxy: activation shift аппроксимирован через embedding shift."""

    z_threshold: float = 2.5
    _embedder: EmbeddingModel | None = None
    _center: np.ndarray | None = None
    _mu: float = 0.0
    _sigma: float = 1.0

    def fit(self, *, embedder: EmbeddingModel, benign_queries: list[str]) -> None:
        self._embedder = embedder
        X = embedder.encode(benign_queries)
        if X.shape[0] == 0:
            self._center = np.zeros((1,), dtype=np.float32)
            self._mu = 0.0
            self._sigma = 1.0
            return
        center = X.mean(axis=0)
        dist = np.linalg.norm(X - center[None, :], axis=1)
        self._center = center
        self._mu = float(dist.mean())
        self._sigma = float(dist.std() + 1e-6)

    def is_attack(self, query: str) -> bool:
        if self._embedder is None or self._center is None:
            return False
        q = self._embedder.encode([query])[0]
        d = float(np.linalg.norm(q - self._center))
        z = (d - self._mu) / self._sigma
        return z > self.z_threshold

