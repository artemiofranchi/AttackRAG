from __future__ import annotations

from dataclasses import dataclass

from attackrag.vector_stores.types import RetrievedChunk


@dataclass
class BGEReranker:
    """Cross-encoder reranker для top-k после retrieval."""

    model_name: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 16

    def __post_init__(self) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        *,
        question: str,
        hits: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not hits:
            return []
        top_k = min(top_k, len(hits))
        pairs = [(question, h.text) for h in hits]
        scores = self._model.predict(pairs, batch_size=self.batch_size)
        scored = list(zip(hits, scores, strict=False))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        return [h for h, _ in scored[:top_k]]

