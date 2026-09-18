from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from attackrag.attacks.metrics import combined_lexical_score, overlap_score
from attackrag.attacks.provenance import collect_provenance
from attackrag.llm import OllamaLLM
from attackrag.llm_caching import CachedLLM
from attackrag.paths import default_golden_qa_path, default_index_dir, repo_root
from attackrag.rag import build_pipeline_from_disk
from attackrag.ragas_eval import load_golden, make_llm_from_env


def _pctl(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = int(round((len(s) - 1) * q))
    return float(s[i])


def main() -> None:
    load_dotenv(repo_root() / ".env")
    p = argparse.ArgumentParser(
        description="Baseline-качество RAG на golden_qa (лексические метрики, latency, provenance; FR-2)."
    )
    p.add_argument("--index", type=Path, default=default_index_dir())
    p.add_argument("--golden", type=Path, default=default_golden_qa_path())
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fast", action="store_true", help="Только первые 10 вопросов (отладка).")
    p.add_argument("--no-cache", action="store_true", help="Не оборачивать LLM в CachedLLM.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = load_golden(args.golden)
    if args.fast:
        rows = rows[:10]

    base_llm = make_llm_from_env()
    llm = base_llm if args.no_cache else CachedLLM(base_llm, role="GENERATOR", model_hint=os.environ.get("OLLAMA_MODEL", "") + os.environ.get("OPENAI_MODEL", ""))
    try:
        pipeline = build_pipeline_from_disk(str(args.index), llm)
        lat: list[float] = []
        per_q: list[dict] = []
        gts = [r["ground_truth"] for r in rows]

        for row in rows:
            q = row["question"]
            t0 = time.perf_counter()
            answer, _ctx, _d = pipeline.query(q)
            lat.append((time.perf_counter() - t0) * 1000.0)
            per_q.append(
                {
                    "id": row.get("id"),
                    "question": q,
                    "answer": answer,
                    "ground_truth": row["ground_truth"],
                    "overlap": overlap_score(answer, row["ground_truth"]),
                    "combined_lexical": combined_lexical_score(answer, row["ground_truth"]),
                }
            )

        overlaps = [x["overlap"] for x in per_q]
        combined = [x["combined_lexical"] for x in per_q]
        prov = collect_provenance(
            index_dir=args.index,
            seed=args.seed,
            rag_config=pipeline.config,
            extra={"benchmark": "baseline", "fast": args.fast, "n_questions": len(rows)},
        )

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = args.out
        if out is None:
            (repo_root() / "runs").mkdir(parents=True, exist_ok=True)
            out = repo_root() / "runs" / f"baseline_{ts}.json"

        payload = {
            "provenance": prov,
            "config": {
                "index": str(args.index.resolve()),
                "golden": str(args.golden.resolve()),
                "fast": args.fast,
                "seed": args.seed,
            },
            "metrics": {
                "mean_overlap": float(mean(overlaps)) if overlaps else 0.0,
                "mean_combined_lexical": float(mean(combined)) if combined else 0.0,
                "latency_ms_p50": _pctl(lat, 0.5),
                "latency_ms_p95": _pctl(lat, 0.95),
                "mean_latency_ms": float(mean(lat)) if lat else 0.0,
            },
            "per_question": per_q,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"out": str(out), **payload["metrics"]}, ensure_ascii=False))
    finally:
        if isinstance(base_llm, OllamaLLM):
            base_llm.close()


if __name__ == "__main__":
    main()
