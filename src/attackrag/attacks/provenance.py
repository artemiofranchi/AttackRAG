from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from attackrag.rag import RAGConfig
from attackrag.vector_stores.meta import read_meta


def _env_snapshot(keys: list[str]) -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in keys}


def collect_provenance(
    *,
    index_dir: Path,
    seed: int,
    rag_config: RAGConfig,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if index_dir.is_dir():
        try:
            meta = read_meta(index_dir)
        except FileNotFoundError:
            meta = {}

    judge_keys = [
        "RAGAS_JUDGE",
        "RAGAS_JUDGE_MODEL",
        "RAGAS_OPENAI_MODEL",
        "RAGAS_GEMINI_MODEL",
        "RAGAS_GEMINI_OPENAI_BASE_URL",
        "RAGAS_EMBEDDING_MODEL",
        "RAGAS_TIMEOUT_SEC",
        "RAGAS_MAX_WORKERS",
        "RAGAS_HTTP_TIMEOUT",
        "RAGAS_JUDGE_HTTP_TIMEOUT",
        "RAGAS_ANSWER_RELEVANCY_STRICTNESS",
    ]
    llm_keys = [
        "LLM_PROVIDER",
        "OLLAMA_MODEL",
        "OLLAMA_HOST",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
        "GEMINI_MODEL",
    ]
    rag_env_keys = [
        "RAG_TOP_K",
        "RAG_CANDIDATE_K",
        "RAG_RERANKER_ENABLED",
        "RAG_RERANKER_MODEL",
        "RAG_RERANKER_BATCH_SIZE",
    ]

    out: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "index_dir": str(index_dir.resolve()),
        "index_meta": meta,
        "rag_config": {
            "embedding_model": rag_config.embedding_model,
            "top_k": rag_config.top_k,
            "candidate_k": rag_config.candidate_k,
            "reranker_enabled": rag_config.reranker_enabled,
            "reranker_model": rag_config.reranker_model,
            "reranker_batch_size": rag_config.reranker_batch_size,
        },
        "env_llm": _env_snapshot(llm_keys),
        "env_rag": _env_snapshot(rag_env_keys),
        "env_ragas_judge": _env_snapshot(judge_keys),
    }
    if extra:
        out["extra"] = extra
    return out
