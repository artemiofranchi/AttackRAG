from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attackrag.embeddings import EmbeddingModel


@dataclass
class LeakSealerDefense:
    """LeakSealer-style proxy: OOD-детектор в embedding space через центроиды."""

    n_centroids: int = 4
    quantile: float = 0.95
    _centroids: np.ndarray | None = None
    _threshold: float | None = None
    _embedder: EmbeddingModel | None = None

    def fit(self, *, embedder: EmbeddingModel, benign_queries: list[str], seed: int = 42) -> None:
        self._embedder = embedder
        X = embedder.encode(benign_queries)
        if X.shape[0] == 0:
            self._centroids = np.zeros((1, 1), dtype=np.float32)
            self._threshold = 0.0
            return
        k = max(1, min(self.n_centroids, X.shape[0]))
        if k == 1:
            centroids = X.mean(axis=0, keepdims=True)
        else:
            # Легковесная инициализация центроидов (без sklearn dependency здесь).
            rng = np.random.default_rng(seed)
            idx = rng.choice(X.shape[0], size=k, replace=False)
            centroids = X[idx].astype(np.float32, copy=True)
        dist = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        dmin = dist.min(axis=1)
        self._centroids = centroids
        self._threshold = float(np.quantile(dmin, self.quantile))

    def is_attack(self, query: str) -> bool:
        if self._embedder is None or self._centroids is None or self._threshold is None:
            return False
        q = self._embedder.encode([query])[0]
        d = np.linalg.norm(self._centroids - q[None, :], axis=1)
        return float(d.min()) > self._threshold

