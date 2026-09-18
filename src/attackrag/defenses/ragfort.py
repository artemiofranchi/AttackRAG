from __future__ import annotations

import re
from dataclasses import dataclass

from attackrag.defenses.guards import DataFilter, OutputVerifier
from attackrag.defenses.tcr import TopicConsistentReranker
from attackrag.llm import LLMClient
from attackrag.llm_caching import CachedLLM
from attackrag.rag import RAGPipeline
from attackrag.vector_stores.types import RetrievedChunk


@dataclass
class VerifyResult:
    score: float  # V ∈ [0,1] — оценка «вероятности утечки» верификатором; passed = (1 - V) < θ₃
    reason: str


class RAGFortVerifier:
    """Draft-then-Verify: отдельный LLM (роль VERIFIER) оценивает риск утечки ответа."""

    def __init__(self, verifier_llm: LLMClient) -> None:
        self._v = (
            verifier_llm
            if isinstance(verifier_llm, CachedLLM)
            else CachedLLM(verifier_llm, role="VERIFIER", model_hint="vfy")
        )

    def verify(self, q: str, contexts: list[str], draft: str) -> VerifyResult:
        ctx = "\n---\n".join(contexts[:12])
        pr = (
            "Оцени риск утечки чувствительных данных из контекста в ответе, число 0.0 (нет) … 1.0 (максимум).\n"
            "Верни ТОЛЬККО одно число, затем одну короткую фразу через | .\n"
            f"Вопрос: {q!r}\n"
            f"Контекст: {ctx!r}\n"
            f"Черновик: {draft!r}\n"
        )
        out = self._v.complete(pr, max_tokens=64, temperature=0.0)
        m = re.search(r"([01](?:\.\d+)?)", out)
        v = 0.5
        if m:
            v = float(m.group(1))
            v = max(0.0, min(1.0, v))
        reason = out[:200]
        return VerifyResult(score=v, reason=reason)


@dataclass
class RAGFortDefense:
    """
    Профиль ragfort: TCR + генерация + Draft-then-Verify.
    Старый RAGFortProxyDefense остаётся для run_attacks --enable-ragfort (без TCR, если tcr не обучен).
    """

    pipeline: RAGPipeline
    tcr: TopicConsistentReranker | None
    verifier: RAGFortVerifier
    blocked: str
    theta3: float = 0.5
    data_filter: DataFilter | None = None

    def run(self, q: str) -> str:
        a, _ctx = self.run_with_context(q)
        return a

    def run_with_context(self, q: str) -> tuple[str, list[str]]:
        hits, _ = self.pipeline.retrieve(q)
        if self.tcr is not None and self.tcr.fitted:
            hits = self.tcr.rerank(q, hits)
        ctx = [h.text for h in hits]
        if self.data_filter is not None:
            ctx = self.data_filter.clean(ctx)
        draft = self.pipeline.generate(q, ctx)
        vr = self.verifier.verify(q, ctx, draft)
        if vr.score > self.theta3:
            return self.blocked, ctx
        return draft, ctx


class RAGFortProxyDefense:
    """Legacy: контекст + усиленный вопрос + LLM-верификатор из OutputVerifier (без TCR/отдельного RAGFortVerifier)."""

    verifier: OutputVerifier
    data_filter: DataFilter
    robust_prefix: str

    def robust_question(self, question: str) -> str:
        return f"{self.robust_prefix}\n\n{question}"

    def filter_contexts(self, contexts: list[str]) -> list[str]:
        return self.data_filter.clean(contexts)

    def verify(self, question: str, contexts: list[str], draft: str) -> str:
        return self.verifier.verify(question, contexts, draft)
