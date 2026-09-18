"""Юнит-тесты без внешних API: IRD, LeakScanner, каскад (лёгкие моки)."""

from __future__ import annotations

import numpy as np
from attackrag.defenses.cascade import CascadeConfig, HARDCascade
from attackrag.defenses.ird import IRDDetector
from attackrag.defenses.output_scanner import LeakScanner
from attackrag.defenses.tcr import TopicConsistentReranker
from attackrag.llm import LLMClient


class _SegStub(LLMClient):
    def __init__(self, frags: list[str]) -> None:
        self._frags = frags

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        import json

        return json.dumps(self._frags, ensure_ascii=False)


def test_ird_zero_one_fragment() -> None:
    def ret(q: str, k: int) -> set[str]:
        return {"a"}

    ird = IRDDetector(ret, _SegStub(["только один"]), k_retr=3, threshold=0.5)
    assert ird.score("x") == 0.0


def test_ird_high_disjoint() -> None:
    def ret(q: str, k: int) -> set[str]:
        return {q.strip()}

    ird = IRDDetector(ret, _SegStub(["фрагмент_α", "фрагмент_β"]), k_retr=3, threshold=0.1)
    assert ird.score("q") > 0.5


class _EmbMini:
    model_name = "test/tiny"

    def encode(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        rng = np.random.default_rng(0)
        return rng.random((n, 8), dtype=np.float32)


def test_tcr_fit_and_rerank() -> None:
    from attackrag.chunking import Chunk
    from attackrag.vector_stores.types import RetrievedChunk

    e = _EmbMini()
    tcr = TopicConsistentReranker(e, k_topics=2)  # type: ignore[arg-type]
    chunks = [Chunk("c0", "d", "текст ноль"), Chunk("c1", "d", "текст один")]
    tcr.fit(chunks, seed=0)
    hits = [
        RetrievedChunk("c0", "d", "текст ноль", 0.2),
        RetrievedChunk("c1", "d", "текст один", 0.3),
    ]
    out = tcr.rerank("вопрос", hits)
    assert len(out) == 2


def test_leak_scanner_regex() -> None:
    s = LeakScanner(regex_patterns=[r"AGRO_API_KEY_[A-Z0-9]{16}"])
    r = s.scan("x AGRO_API_KEY_ABCDEFGHIJKLMNOP r", ["ctx"])
    assert r["hit"] is True


def test_cascade_blocks_high_ird() -> None:
    def ret_ids(q: str, k: int) -> set[str]:
        return {q.strip()}

    ird = IRDDetector(ret_ids, _SegStub(["A", "B"]), k_retr=2, threshold=0.01)

    class _TCR:
        fitted = False

        def rerank(self, q: str, hits):
            return hits

        def context_anom(self, hits) -> float:
            return 0.0

        def anom_score(self, _cid: str) -> float:
            return 0.0

    class _Ver:
        def verify(self, q: str, contexts: list[str], draft: str):
            from attackrag.defenses.ragfort import VerifyResult

            return VerifyResult(score=0.0, reason="ok")

    class _Pipe:
        def retrieve(self, q: str):
            from attackrag.vector_stores.types import RetrievedChunk

            return [RetrievedChunk("c", "d", "t", 0.1)], np.zeros(3, dtype=np.float32)

        def generate(self, q: str, ctx: list[str]) -> str:
            return "y"

    out = HARDCascade(
        CascadeConfig(
            theta1_0=0.01,
            enable_stage2=False,
            enable_stage3=False,
            enable_stage4=False,
            adaptive=False,
        ),
        ird,
        _TCR(),  # type: ignore[arg-type]
        _Ver(),  # type: ignore[arg-type]
        LeakScanner(regex_patterns=[]),
        _Pipe(),  # type: ignore[arg-type]
    ).query("z")
    assert out["blocked_at_stage"] == 1


def test_cascade_robust_prefix_applies_only_to_generate() -> None:
    """IRD/retrieve/TCR получают чистый q; префикс — только в generate (см. profile hard)."""
    from attackrag.vector_stores.types import RetrievedChunk

    def ret_ids(_q: str, k: int) -> set[str]:
        return {"same"}

    ird = IRDDetector(ret_ids, _SegStub(["a", "b"]), k_retr=2, threshold=0.99)

    class _TCR:
        fitted = False

        def rerank(self, q: str, hits):
            return hits

        def context_anom(self, hits) -> float:
            return 0.0

        def anom_score(self, _cid: str) -> float:
            return 0.0

    class _Ver:
        def verify(self, q: str, contexts: list[str], draft: str):
            from attackrag.defenses.ragfort import VerifyResult

            return VerifyResult(score=0.0, reason="ok")

    retrieve_q: list[str] = []
    generate_q: list[str] = []

    class _Pipe:
        def retrieve(self, q: str):
            retrieve_q.append(q)
            return [RetrievedChunk("c", "d", "t", 0.1)], np.zeros(3, dtype=np.float32)

        def generate(self, q: str, ctx: list[str]) -> str:
            generate_q.append(q)
            return "y"

    pipe = _Pipe()  # type: ignore[arg-type]
    HARDCascade(
        CascadeConfig(
            theta1_0=0.99,
            enable_stage2=False,
            enable_stage3=False,
            enable_stage4=False,
            adaptive=False,
        ),
        ird,
        _TCR(),  # type: ignore[arg-type]
        _Ver(),  # type: ignore[arg-type]
        LeakScanner(regex_patterns=[]),
        pipe,
        robust_prefix="ROBUST_LINE",
    ).query("user_question")

    assert retrieve_q == ["user_question"]
    assert len(generate_q) == 1
    assert generate_q[0].startswith("ROBUST_LINE")
    assert "user_question" in generate_q[0]
