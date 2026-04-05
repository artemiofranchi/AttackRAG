from __future__ import annotations

import argparse
import json
from pathlib import Path


def _collect_rows(files: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in files:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics = payload.get("metrics") or {}
        config = payload.get("config") or {}
        rows.append(
            {
                "file": p.name,
                "attack": str(config.get("attack") or "unknown"),
                "asr": metrics.get("asr"),
                "fpr": metrics.get("fpr"),
                "bpd": metrics.get("bpd"),
                "latency_overhead_ms": metrics.get("latency_overhead_ms"),
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Собрать CSV и графики по runs/attack_*.json")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("runs"),
        help="Каталог с attack_*.json",
    )
    p.add_argument(
        "--pattern",
        default="attack_*.json",
        help="Шаблон файлов для агрегации",
    )
    p.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("runs/attack_summary"),
        help="Префикс выходных файлов (без расширения)",
    )
    args = p.parse_args()

    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"Файлы не найдены: {args.input_dir}/{args.pattern}")

    rows = _collect_rows(files)
    if not rows:
        raise SystemExit("Не удалось прочитать ни одного корректного attack JSON")

    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows).sort_values(["attack", "file"])
    csv_path = args.out_prefix.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    metric_names = ["asr", "fpr", "bpd", "latency_overhead_ms"]
    for m in metric_names:
        if m not in df.columns:
            continue
        grouped = df.groupby("attack", dropna=False)[m].mean()
        ax = grouped.plot(kind="bar", legend=False, title=f"{m} by attack", figsize=(6, 4))
        ax.set_ylabel(m)
        plt.tight_layout()
        out_png = args.out_prefix.parent / f"{args.out_prefix.name}_{m}.png"
        plt.savefig(out_png, dpi=160)
        plt.close()

    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "count_files": len(files),
                "metrics": metric_names,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

