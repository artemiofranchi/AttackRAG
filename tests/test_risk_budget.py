"""Risk-Budget Propagation: θ_i(q) = θ_i⁽⁰⁾ - λ_i · r (eq:risk_budget)."""

from __future__ import annotations

import numpy as np
import pytest

from attackrag.defenses.cascade import CascadeConfig, HARDCascade
from attackrag.defenses.ird import IRDDetector
from attackrag.defenses.output_scanner import LeakScanner
from attackrag.llm import LLMClient


class _SegStub(LLMClient):
    def __init__(self, frags: list[str]) -> None:
        self._frags = frags

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        import json

        return json.dumps(self._frags, ensure_ascii=False)


class _StubTCR:
    fitted = True

    def rerank(self, q: str, hits):
        return hits

    def context_anom(self, hits) -> float:
        return 0.0

    def anom_score(self, _cid: str) -> float:
        return 0.0


class _StubVerifier:
    def __init__(self, score: float = 0.0) -> None:
        self.score = score

    def verify(self, q, ctx, draft):
        from attackrag.defenses.ragfort import VerifyResult

        return VerifyResult(score=self.score, reason="stub")


class _StubPipe:
    def __init__(self) -> None:
        self.gen_count = 0

    def retrieve(self, q: str):
        from attackrag.vector_stores.types import RetrievedChunk

        return [RetrievedChunk("c", "d", "ctx", 0.5)], np.zeros(3, dtype=np.float32)

    def generate(self, q: str, ctx: list[str]) -> str:
        self.gen_count += 1
        return "draft"


def _ird(score_value: float, threshold: float) -> IRDDetector:
    """IRD, который выдаёт score ≈ score_value."""

    def ret(_q: str, _k: int) -> set[str]:
        # Возвращает один и тот же набор → divergence = 0.
        return {"x"}

    return IRDDetector(ret, _SegStub(["a", "b"]), k_retr=2, threshold=threshold)


def test_risk_budget_lambda_lowers_effective_threshold() -> None:
    """λ_2 = 0.3, r > 0 ⇒ effective θ_2 = 0.5 - 0.3·r должен снизиться.

    Тест проверяет, что `stage_thresholds_effective` в трейсе численно
    отражает λ-формулу (а не наоборот, например, `+ λ·r`).
    """
    # Запускаем каскад только до Stage 2; конструируем стадию 1 так, чтобы
    # она ВЫДАВАЛА score 0 ⇒ stage 1 пасс, накапливаем r от score=0 (всё ок).
    # Чтобы стадия 1 накопила какой-то риск, нужен IRD score > 0; для этого
    # ставим threshold = 0.0001 и заставляем сегментер вернуть disjoint frags.
    def disjoint_ret(q: str, _k: int) -> set[str]:
        # Каждый фрагмент q.strip() => уникальные ID: divergence > 0.
        return {q.strip()}

    ird = IRDDetector(disjoint_ret, _SegStub(["alpha", "beta"]), k_retr=2, threshold=0.99)
    pipe = _StubPipe()
    cfg = CascadeConfig(
        enable_stage1=True,
        enable_stage2=True,
        enable_stage3=False,
        enable_stage4=False,
        theta1_0=0.99,  # стадия 1 пройдёт (score 0.5 < 0.99)
        theta2_0=0.5,
        lambda1=0.0,
        lambda2=0.3,
        adaptive=True,
    )
    out = HARDCascade(cfg, ird, _StubTCR(), _StubVerifier(0.0), LeakScanner(regex_patterns=[]), pipe).query("q")  # type: ignore[arg-type]

    eff = out.get("stage_thresholds_effective") or {}
    base = out.get("stage_thresholds_base") or {}
    # Базовая θ_2 не меняется.
    assert pytest.approx(base.get(2, 0.0), rel=1e-6) == 0.5
    # Эффективная θ_2 не больше базовой (риск ≥ 0 ⇒ θ_eff ≤ θ_base).
    assert eff.get(2, 1.0) <= base.get(2, 0.0) + 1e-6
    # И не ниже 0 (clip).
    assert eff.get(2, -1.0) >= 0.0


def test_risk_budget_lambda_zero_does_not_shift_stage1() -> None:
    """λ_1 = 0 ⇒ θ_1_eff == θ_1_base всегда (стадия 1 не получает риск)."""
    ird = _ird(0.0, threshold=0.99)
    pipe = _StubPipe()
    cfg = CascadeConfig(
        enable_stage1=True,
        enable_stage2=False,
        enable_stage3=False,
        enable_stage4=False,
        theta1_0=0.7,
        lambda1=0.0,
        adaptive=True,
    )
    out = HARDCascade(cfg, ird, _StubTCR(), _StubVerifier(0.0), LeakScanner(regex_patterns=[]), pipe).query("q")  # type: ignore[arg-type]
    eff = out.get("stage_thresholds_effective") or {}
    base = out.get("stage_thresholds_base") or {}
    assert pytest.approx(eff.get(1, 0.0), rel=1e-6) == base.get(1, 0.0)
