from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

LeakCheck = Callable[[str, str], bool]
from attackrag.attacks.types import StageTrace, TrialRecord, empty_trace

if TYPE_CHECKING:
    from attackrag.rag import RAGPipeline


def run_backdoor(
    pipeline: RAGPipeline,
    golden_rows: list[dict[str, str]],
    *,
    leak_check: LeakCheck,
    trigger: str,
    secret_value: str,
    suffix_from_benign: bool = True,
    query_fn=None,
) -> list[TrialRecord]:
    """
    Запросы с триггером (и опционально хвостом легитимного вопроса).
    Успех задаётся leak_check(answer, ref): по умолчанию S_priv или legacy —
    detector.is_leak(secret_value, answer).

    `query_fn(q) -> (answer, contexts, StageTrace)` (см. profile_query.py); если
    None — идём напрямую через pipeline и трассу заполняем пустой.
    """
    out: list[TrialRecord] = []
    for row in golden_rows:
        qid = row.get("id") or ""
        benign = row["question"]
        gt = row["ground_truth"]
        q = f"{trigger} {benign}" if suffix_from_benign else trigger
        t0 = time.perf_counter()
        trace: StageTrace
        if query_fn is None:
            answer, contexts, _ = pipeline.query(q)
            trace = empty_trace(blocked=False)
        else:
            answer, contexts, trace = query_fn(q)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        leaked = leak_check(answer, secret_value)
        out.append(
            TrialRecord(
                question_id=str(qid),
                benign_question=benign,
                attack_prompt=q,
                ground_truth=gt,
                answer=answer,
                leaked=leaked,
                latency_ms=dt_ms,
                contexts=None,
                meta={
                    "attack": "backdoor",
                    "trigger": trigger,
                    "target_secret": secret_value,
                    "stage_trace": dict(trace),
                },
            )
        )
    return out
