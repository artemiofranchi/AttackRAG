from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset

from attackrag.llm import GeminiLLM, LLMClient, OllamaLLM, OpenAICompatLLM
from attackrag.paths import default_golden_qa_path
from attackrag.rag import RAGPipeline, build_pipeline_from_disk


class _LegacyEmbeddingShim:
    """Метрика answer_relevancy в RAGAS зовёт embed_query / embed_documents (как LangChain).

    Современный ragas.embeddings.OpenAIEmbeddings даёт только embed_text / embed_texts.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_texts(texts)


def _shim_embeddings_for_ragas(embeddings: Any) -> Any:
    if embeddings is None:
        return embeddings
    if hasattr(embeddings, "embed_query") and hasattr(embeddings, "embed_documents"):
        return embeddings
    if hasattr(embeddings, "embed_text") and hasattr(embeddings, "embed_texts"):
        return _LegacyEmbeddingShim(embeddings)
    return embeddings


def _http_timeout_seconds() -> float:
    """Таймаут одного HTTP-запроса к API судьи (OpenAI SDK / LangChain)."""
    raw = os.environ.get("RAGAS_HTTP_TIMEOUT") or os.environ.get("RAGAS_JUDGE_HTTP_TIMEOUT")
    if raw is None or str(raw).strip() == "":
        return 300.0
    return float(raw)


def _ragas_run_config() -> Any:
    """Таймаут на одну ячейку metric×строка и пул воркеров (иначе Gemini + 4 метрики ловят TimeoutError)."""
    from ragas.run_config import RunConfig

    t_raw = os.environ.get("RAGAS_TIMEOUT_SEC")
    timeout = int(t_raw) if t_raw and str(t_raw).strip() else 600
    w_raw = os.environ.get("RAGAS_MAX_WORKERS")
    max_workers = int(w_raw) if w_raw and str(w_raw).strip() else 4
    return RunConfig(timeout=timeout, max_workers=max_workers)


def _ragas_judge_chat_openai(
    *,
    model: str,
    api_key: str,
    base_url: str | None,
) -> Any:
    """Судья для evaluate(): «голый» ChatOpenAI — RAGAS сам оборачивает в LangchainLLMWrapper(run_config=...).

    Не оборачивать здесь в LangchainLLMWrapper второй раз.
    """
    from langchain_openai import ChatOpenAI

    kw: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "timeout": _http_timeout_seconds(),
        "temperature": 0.01,
    }
    if base_url:
        kw["base_url"] = base_url.rstrip("/") + "/"
    return ChatOpenAI(**kw)


def load_golden(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("golden_qa.json должен быть JSON-массивом объектов")
    return data


def make_llm_from_env() -> LLMClient:
    provider = (os.environ.get("LLM_PROVIDER") or "openai").lower().strip()
    if provider in ("ollama", "llama", "local"):
        return OllamaLLM(
            model=os.environ.get("OLLAMA_MODEL") or "llama3.1",
            host=os.environ.get("OLLAMA_HOST"),
        )
    if provider in ("gemini", "google"):
        return GeminiLLM(
            model=os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash",
        )
    return OpenAICompatLLM(
        model=os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )


def _ragas_eval_backend_kwargs() -> dict[str, Any]:
    """RAGAS по умолчанию создаёт OpenAI(); для Gemini/Ollama передаём llm и embeddings явно."""
    judge = (
        os.environ.get("RAGAS_JUDGE")
        or os.environ.get("LLM_PROVIDER")
        or "openai"
    ).lower().strip()
    if judge in ("ollama", "llama", "local"):
        from openai import OpenAI
        from ragas.embeddings import OpenAIEmbeddings

        host = (os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        api_key = os.environ.get("OLLAMA_API_KEY") or "ollama"
        base = f"{host}/v1"
        client = OpenAI(base_url=base, api_key=api_key, timeout=_http_timeout_seconds())
        chat_model = os.environ.get("OLLAMA_MODEL") or "llama3.1"
        emb_model = os.environ.get("OLLAMA_EMBEDDING_MODEL") or "nomic-embed-text"
        return {
            "llm": _ragas_judge_chat_openai(
                model=chat_model,
                api_key=api_key,
                base_url=base,
            ),
            "embeddings": OpenAIEmbeddings(client=client, model=emb_model),
        }
    if judge in ("gemini", "google"):
        # Судья: LangChain ChatOpenAI на OpenAI-совместимом endpoint Gemini — иначе Instructor
        # даёт одну генерацию, а answer_relevancy просит strictness=3 (n=3).
        # См. https://ai.google.dev/gemini-api/docs/openai
        from openai import OpenAI
        from ragas.embeddings import OpenAIEmbeddings

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "Для RAGAS с Gemini задайте GEMINI_API_KEY или GOOGLE_API_KEY "
                "(или RAGAS_JUDGE=openai и OPENAI_API_KEY)."
            )
        base_url = (
            os.environ.get("RAGAS_GEMINI_OPENAI_BASE_URL")
            or os.environ.get("GEMINI_OPENAI_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        client = OpenAI(api_key=key, base_url=base_url, timeout=_http_timeout_seconds())
        model = (
            os.environ.get("RAGAS_GEMINI_MODEL")
            or os.environ.get("GEMINI_MODEL")
            or "gemini-2.0-flash"
        )
        emb_model = (
            os.environ.get("RAGAS_GEMINI_EMBEDDING_MODEL")
            or os.environ.get("GEMINI_EMBEDDING_MODEL")
            or "gemini-embedding-001"
        )
        return {
            "llm": _ragas_judge_chat_openai(
                model=model,
                api_key=key,
                base_url=base_url,
            ),
            "embeddings": OpenAIEmbeddings(client=client, model=emb_model),
        }
    if judge in ("openai",):
        from openai import OpenAI
        from ragas.embeddings import OpenAIEmbeddings

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "RAGAS_JUDGE=openai требует OPENAI_API_KEY "
                "(или выберите RAGAS_JUDGE=ollama / gemini)."
            )
        openai_base = os.environ.get("OPENAI_BASE_URL")
        client = OpenAI(
            api_key=key,
            base_url=openai_base,
            timeout=_http_timeout_seconds(),
        )
        judge_model = (
            os.environ.get("RAGAS_JUDGE_MODEL")
            or os.environ.get("RAGAS_OPENAI_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        emb_model = os.environ.get("RAGAS_EMBEDDING_MODEL") or "text-embedding-3-small"
        return {
            "llm": _ragas_judge_chat_openai(
                model=judge_model,
                api_key=key,
                base_url=openai_base,
            ),
            "embeddings": OpenAIEmbeddings(client=client, model=emb_model),
        }
    return {}


def run_rag_over_golden(
    pipeline: RAGPipeline,
    golden_path: Path,
) -> dict[str, list[Any]]:
    items = load_golden(golden_path)
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []
    for row in items:
        q = row["question"]
        gt = row["ground_truth"]
        a, ctx, _ = pipeline.query(q)
        questions.append(q)
        answers.append(a)
        contexts.append(ctx)
        ground_truths.append(gt)
    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def evaluate_with_ragas(bundle: dict[str, list[Any]]) -> Any:
    from ragas import evaluate
    # NOTE:
    # In some ragas versions, metrics under ragas.metrics.collections are
    # classes/modules that must be manually initialized with llm/embeddings.
    # The legacy imports below provide ready-to-use metric objects and keep
    # this CLI path stable across versions.
    from ragas.metrics import (
        AnswerRelevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    strictness_raw = os.environ.get("RAGAS_ANSWER_RELEVANCY_STRICTNESS")
    strictness = int(strictness_raw) if strictness_raw and str(strictness_raw).strip() else 3

    ds = Dataset.from_dict(bundle)
    extra = _ragas_eval_backend_kwargs()
    if "embeddings" in extra and extra["embeddings"] is not None:
        extra["embeddings"] = _shim_embeddings_for_ragas(extra["embeddings"])
    return evaluate(
        ds,
        metrics=[
            faithfulness,
            AnswerRelevancy(strictness=strictness),
            context_precision,
            context_recall,
        ],
        run_config=_ragas_run_config(),
        **extra,
    )


def default_run(
    *,
    index_dir: Path,
    golden_path: Path | None = None,
) -> Any:
    golden_path = golden_path or default_golden_qa_path()
    llm = make_llm_from_env()
    try:
        pipeline = build_pipeline_from_disk(str(index_dir), llm)
        bundle = run_rag_over_golden(pipeline, golden_path)
        return evaluate_with_ragas(bundle)
    finally:
        if isinstance(llm, OllamaLLM):
            llm.close()
