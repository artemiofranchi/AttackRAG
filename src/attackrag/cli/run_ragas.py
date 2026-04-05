from __future__ import annotations

import argparse
import json
from pathlib import Path
import typing as t

from dotenv import load_dotenv

from attackrag.llm import OllamaLLM
from attackrag.paths import default_golden_qa_path, default_index_dir, repo_root
from attackrag.rag import build_pipeline_from_disk
from attackrag.ragas_eval import evaluate_with_ragas, make_llm_from_env, run_rag_over_golden


def _to_jsonable(obj: t.Any) -> t.Any:
    # primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # common ragas wrapper: MetricResult
    if hasattr(obj, "value"):
        try:
            return _to_jsonable(obj.value)
        except Exception:
            pass

    # objects that can describe themselves
    for meth in ("to_dict", "model_dump", "dict", "asdict"):
        f = getattr(obj, meth, None)
        if callable(f):
            try:
                return _to_jsonable(f())
            except Exception:
                pass

    # mapping-like
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}

    # fallback: try to cast to dict
    try:
        d = dict(obj)
        return {k: _to_jsonable(v) for k, v in d.items()}
    except Exception:
        return str(obj)


def _serialize_evaluation_result(result: t.Any) -> t.Any:
    return _to_jsonable(result)


def main() -> None:
    load_dotenv(repo_root() / ".env")
    p = argparse.ArgumentParser(description="Прогнать RAG по golden_qa и посчитать метрики RAGAS")
    p.add_argument("--index", type=Path, default=default_index_dir())
    p.add_argument("--golden", type=Path, default=default_golden_qa_path())
    p.add_argument(
        "--export-json",
        type=Path,
        default=None,
        help="Сохранить сырые ответы и контексты в JSON (для отладки)",
    )
    p.add_argument(
        "--dry-ragas",
        action="store_true",
        help="Только сгенерировать ответы без вызова RAGAS (без LLM-судей)",
    )
    args = p.parse_args()

    llm = make_llm_from_env()
    try:
        pipeline = build_pipeline_from_disk(str(args.index), llm)
        bundle = run_rag_over_golden(pipeline, args.golden)
    finally:
        if isinstance(llm, OllamaLLM):
            llm.close()

    if args.dry_ragas:
        print(json.dumps(bundle, ensure_ascii=False, indent=2))
        if args.export_json is not None:
            args.export_json.parent.mkdir(parents=True, exist_ok=True)
            args.export_json.write_text(
                json.dumps({"bundle": bundle, "metrics": None}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return

    result = evaluate_with_ragas(bundle)
    print(result)

    if args.export_json is not None:
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        metrics_payload = _serialize_evaluation_result(result)
        args.export_json.write_text(
            json.dumps(
                {"bundle": bundle, "metrics": metrics_payload},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
