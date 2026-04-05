from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from attackrag.embeddings import EmbeddingModel
from attackrag.llm import LLMClient
from attackrag.reranking import BGEReranker
from attackrag.vector_stores import load_vector_store
from attackrag.vector_stores.protocol import VectorStore
from attackrag.vector_stores.types import RetrievedChunk


@dataclass
class RAGConfig:
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 8
    candidate_k: int = 24
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_batch_size: int = 16


class RAGPipeline:
    """f_RAG(q, D): retrieval по индексу D, затем генерация G(T(q, C))."""

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingModel,
        llm: LLMClient,
        config: RAGConfig | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._llm = llm
        self._config = config or RAGConfig()
        self._reranker = (
            BGEReranker(
                model_name=self._config.reranker_model,
                batch_size=self._config.reranker_batch_size,
            )
            if self._config.reranker_enabled
            else None
        )

    @property
    def config(self) -> RAGConfig:
        return self._config

    def _template(self, question: str, contexts: list[str]) -> str:
        ctx = "\n\n---\n\n".join(contexts)
        return (
            "Контекст (фрагменты из корпуса):\n"
            f"{ctx}\n\n"
            f"Вопрос: {question}\n\n"
            "Ответ:"
        )

    def retrieve(self, question: str) -> tuple[list[RetrievedChunk], np.ndarray]:
        q_emb = self._embedder.encode([question])[0]
        retrieve_k = (
            max(self._config.top_k, self._config.candidate_k)
            if self._config.reranker_enabled
            else self._config.top_k
        )
        hits = self._store.search(q_emb, retrieve_k)
        if self._reranker is not None:
            hits = self._reranker.rerank(
                question=question,
                hits=hits,
                top_k=self._config.top_k,
            )
        else:
            hits = hits[: self._config.top_k]
        return hits, q_emb

    def generate(self, question: str, contexts: list[str]) -> str:
        prompt = self._template(question, contexts)
        return self._llm.complete(prompt)

    def query(self, question: str) -> tuple[str, list[str], np.ndarray]:
        hits, q_emb = self.retrieve(question)
        contexts = [h.text for h in hits]
        answer = self.generate(question, contexts)
        return answer, contexts, q_emb


def build_pipeline_from_disk(
    index_dir: str,
    llm: LLMClient,
    config: RAGConfig | None = None,
) -> RAGPipeline:
    store = load_vector_store(Path(index_dir))
    base = config or RAGConfig()
    env_top_k = os.environ.get("RAG_TOP_K")
    env_candidate_k = os.environ.get("RAG_CANDIDATE_K")
    env_rerank = (os.environ.get("RAG_RERANKER_ENABLED") or "").strip().lower()
    env_rerank_model = os.environ.get("RAG_RERANKER_MODEL")
    env_rerank_bs = os.environ.get("RAG_RERANKER_BATCH_SIZE")

    top_k = int(env_top_k) if env_top_k else base.top_k
    candidate_k = int(env_candidate_k) if env_candidate_k else base.candidate_k
    reranker_enabled = (
        env_rerank in {"1", "true", "yes", "on"} if env_rerank else base.reranker_enabled
    )
    reranker_model = env_rerank_model or base.reranker_model
    reranker_batch_size = int(env_rerank_bs) if env_rerank_bs else base.reranker_batch_size

    cfg = RAGConfig(
        embedding_model=store.embedding_model or base.embedding_model,
        top_k=top_k,
        candidate_k=candidate_k,
        reranker_enabled=reranker_enabled,
        reranker_model=reranker_model,
        reranker_batch_size=reranker_batch_size,
    )
    embedder = EmbeddingModel(cfg.embedding_model)
    return RAGPipeline(store=store, embedder=embedder, llm=llm, config=cfg)
