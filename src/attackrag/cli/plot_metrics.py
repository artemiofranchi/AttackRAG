from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_metrics(data: dict[str, Any]) -> dict[str, Any] | None:
    # New format: {"bundle": ..., "metrics": {...}}
    if "metrics" in data:
        metrics = data.get("metrics")
        if isinstance(metrics, dict):
            return metrics
        # Compatibility: metrics could be stored as repr-string:
        # "{'faithfulness': 0.9, ...}"
        if isinstance(metrics, str):
            try:
                parsed = ast.literal_eval(metrics)
                if isinstance(parsed, dict):
                    return parsed
            except (SyntaxError, ValueError):
                return None
        return None
    # Old format: bundle only (no metrics export)
    return None


def _format_value(v: Any) -> tuple[float | None, bool]:
    """
    Returns (value_as_float_or_None, is_missing_or_nan).
    """
    if v is None:
        return None, True
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None, True
    if math.isnan(fv):
        return None, True
    return fv, False


def main() -> None:
    p = argparse.ArgumentParser(description="Построить PNG-график RAGAS метрик")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("runs") / "rag_bundle.json",
        help="JSON файл, который создаёт attack-rag-run-ragas --export-json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs") / "ragas_metrics.png",
        help="Куда сохранить PNG",
    )
    args = p.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Нет файла: {args.input}")

    data = _load_json(args.input)
    metrics = _extract_metrics(data)
    if not metrics:
        raise RuntimeError(
            "В JSON нет поля 'metrics'. "
            "Сначала пересобери запуск: attack-rag-run-ragas --export-json runs\\rag_bundle.json"
        )

    keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer relevancy",
        "context_precision": "Context precision",
        "context_recall": "Context recall",
    }

    xs: list[str] = []
    ys: list[float] = []
    annotations: list[str] = []
    for k in keys:
        v = metrics.get(k)
        fv, missing = _format_value(v)
        xs.append(labels.get(k, k))
        if fv is None:
            ys.append(0.0)
            annotations.append("NA" if missing else "")
        else:
            ys.append(fv)
            # показываем 3 знака после точки, чтобы в диссере читалось
            annotations.append(f"{fv:.3f}")

    # Импорт matplotlib после проверок, чтобы не требовать extra для тех случаев,
    # когда metrics в файле отсутствуют.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(xs, ys, color="#4C78A8")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("RAGAS metrics (mean over golden QA)")

    for bar, txt in zip(bars, annotations, strict=False):
        height = bar.get_height()
        if txt and txt != "NA":
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                txt,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        elif txt == "NA":
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.02,
                "NA",
                ha="center",
                va="bottom",
                fontsize=9,
                color="gray",
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

