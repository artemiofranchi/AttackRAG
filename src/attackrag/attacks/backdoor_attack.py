from __future__ import annotations

import time
from typing import TYPE_CHECKING

from attackrag.attacks.detectors import LeakDetector
from attackrag.attacks.types import TrialRecord

if TYPE_CHECKING:
    from attackrag.rag import RAGPipeline


def run_backdoor(
    pipeline: RAGPipeline,
    golden_rows: list[dict[str, str]],
    *,
    detector: LeakDetector,
    trigger: str,
    secret_value: str,
    suffix_from_benign: bool = True,
    query_fn=None,
) -> list[TrialRecord]:
    """
    Запросы с триггером (и опционально хвостом легитимного вопроса).
    Успех — утечка secret_value (не ground_truth), индекс должен быть собран с poison-документом.
    """
    out: list[TrialRecord] = []
    for row in golden_rows:
        qid = row.get("id") or ""
        benign = row["question"]
        gt = row["ground_truth"]
        q = f"{trigger} {benign}" if suffix_from_benign else trigger
        t0 = time.perf_counter()
        if query_fn is None:
            answer, contexts, _ = pipeline.query(q)
        else:
            answer, contexts = query_fn(q)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        leaked = detector.is_leak(secret_value, answer)
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
                },
            )
        )
    return out
