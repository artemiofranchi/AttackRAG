"""HARDCascade: IRD → TCR → Draft+Verify → LeakScanner + Risk-Budget (eq:risk_budget)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from attackrag.defenses.ird import IRDDetector
from attackrag.defenses.output_scanner import LeakScanner
from attackrag.defenses.ragfort import RAGFortVerifier
from attackrag.defenses.tcr import TopicConsistentReranker
from attackrag.rag import RAGPipeline


@dataclass
class CascadeConfig:
    enable_stage1: bool = True
    enable_stage2: bool = True
    enable_stage3: bool = True
    enable_stage4: bool = True
    theta1_0: float = 0.6
    theta2_0: float = 0.5
    theta3_0: float = 0.5
    theta4_0: float = 0.5
    lambda1: float = 0.0
    lambda2: float = 0.3
    lambda3: float = 0.3
    lambda4: float = 0.3
    adaptive: bool = True


class HARDCascade:
    def __init__(
        self,
        config: CascadeConfig,
        ird: IRDDetector,
        tcr: TopicConsistentReranker,
        verifier: RAGFortVerifier,
        scanner: LeakScanner,
        pipeline: RAGPipeline,
        *,
        blocked_text: str = "Content blocked by HARD policy",
    ) -> None:
        self._cfg = config
        self._ird = ird
        self._tcr = tcr
        self._verifier = verifier
        self._scanner = scanner
        self._pipe = pipeline
        self._blocked = blocked_text

    def query(self, q: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        cfg = self._cfg
        r_acc = 0.0
        eff: dict[int, float] = {}
        raw_t = {
            1: cfg.theta1_0,
            2: cfg.theta2_0,
            3: cfg.theta3_0,
            4: cfg.theta4_0,
        }

        def eff_theta(i: int) -> float:
            lam = [cfg.lambda1, cfg.lambda2, cfg.lambda3, cfg.lambda4][i - 1]
            base = raw_t[i]
            if not cfg.adaptive:
                return base
            return float(np.clip(base - lam * r_acc, 0.01, 1.0))

        h1 = self._ird.score(q) if cfg.enable_stage1 else 0.0
        t1 = eff_theta(1)
        eff[1] = t1
        sp1 = True if not cfg.enable_stage1 else (h1 < t1)
        if cfg.enable_stage1 and not sp1:
            for i in (2, 3, 4):
                eff[i] = eff_theta(i)
            return self._fin(
                t0,
                blocked=1,
                answer=self._blocked,
                h1=h1,
                h2=0.0,
                h3=0.0,
                h4=0.0,
                sp={1: False, 2: True, 3: True, 4: True},
                eff=eff,
                draft=None,
                ctx=[],
            )
        if cfg.adaptive and cfg.enable_stage1 and sp1:
            r_acc += min(h1 / max(t1, 1e-6), 1.0)

        hits, _ = self._pipe.retrieve(q)
        if self._tcr.fitted and cfg.enable_stage2:
            hits = self._tcr.rerank(q, hits)
            t2 = eff_theta(2)
            eff[2] = t2
            hits_f = [h for h in hits if self._tcr.anom_score(h.chunk_id) <= t2]
            if not hits_f and hits:
                hits_f = hits[:1]
            hits = hits_f
            h2 = self._tcr.context_anom(hits)
        else:
            t2 = eff_theta(2)
            eff[2] = t2
            h2 = 0.0
        sp2 = True if not cfg.enable_stage2 else (h2 < t2)
        if cfg.adaptive and cfg.enable_stage2:
            r_acc += min(h2 / max(t2, 1e-6), 1.0)

        ctx = [h.text for h in hits]
        draft = self._pipe.generate(q, ctx)
        if cfg.enable_stage3:
            vr = self._verifier.verify(q, ctx, draft)
            h3 = float(vr.score)
        else:
            h3 = 0.0
        t3 = eff_theta(3)
        eff[3] = t3
        sp3 = True if not cfg.enable_stage3 else (h3 < t3)
        if cfg.enable_stage3 and not sp3:
            return self._fin(
                t0,
                blocked=3,
                answer=self._blocked,
                h1=h1,
                h2=h2,
                h3=h3,
                h4=0.0,
                sp={1: sp1, 2: sp2, 3: False, 4: True},
                eff=eff,
                draft=draft,
                ctx=ctx,
            )
        if cfg.adaptive and cfg.enable_stage3 and sp3:
            r_acc += min(h3 / max(t3, 1e-6), 1.0)

        if cfg.enable_stage4:
            scan = self._scanner.scan(draft, ctx)
            h4 = 1.0 if scan["hit"] else 0.0
        else:
            h4 = 0.0
        t4 = eff_theta(4)
        eff[4] = t4
        sp4 = True if not cfg.enable_stage4 else (h4 < t4)
        if cfg.enable_stage4 and not sp4:
            return self._fin(
                t0,
                blocked=4,
                answer=self._blocked,
                h1=h1,
                h2=h2,
                h3=h3,
                h4=h4,
                sp={1: sp1, 2: sp2, 3: sp3, 4: False},
                eff=eff,
                draft=draft,
                ctx=ctx,
            )

        return self._fin(
            t0,
            blocked=None,
            answer=draft,
            h1=h1,
            h2=h2,
            h3=h3,
            h4=h4,
            sp={1: sp1, 2: sp2, 3: sp3, 4: sp4},
            eff=eff,
            draft=draft,
            ctx=ctx,
        )

    def _fin(
        self,
        t0: float,
        *,
        blocked: int | None,
        answer: str,
        h1: float,
        h2: float,
        h3: float,
        h4: float,
        sp: dict[int, bool],
        eff: dict[int, float],
        draft: str | None,
        ctx: list[str],
    ) -> dict[str, Any]:
        return {
            "answer": answer,
            "blocked_at_stage": blocked,
            "stage_scores": {1: h1, 2: h2, 3: h3, 4: h4},
            "stage_thresholds_effective": eff,
            "stage_thresholds_base": {
                1: self._cfg.theta1_0,
                2: self._cfg.theta2_0,
                3: self._cfg.theta3_0,
                4: self._cfg.theta4_0,
            },
            "stage_passed": sp,
            "contexts": ctx,
            "draft": draft,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }
