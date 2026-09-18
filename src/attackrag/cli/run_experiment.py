"""
Универсальный оркестратор эксперимента (Шаг 6 плана + FR-3..FR-9 формализации).

Структура артефактов (см. план):

    runs/experiment_<ts>/
    ├── seed_<N>/
    │   └── profile_<name>/
    │       ├── attack_pi/report.json
    │       ├── attack_backdoor/report.json
    │       └── attack_secret/report.json
    ├── summary.json     # сводная таблица (profile × attack), агрегированная по seed-ам
    ├── summary.csv      # то же в CSV
    ├── provenance.json  # git-коммит, env (без секретов), seeds, profiles, attacks
    ├── config.yaml      # эффективная конфигурация прогона
    └── logs/<seed>__<profile>__<attack>.log

Режимы:
  smoke   — все профили × все атаки × 1 сид × 5 вопросов × 1 iter SECRET (UI/CLI)
  fast    — все профили × все атаки × 1 сид × 10 вопросов × 3 iter SECRET (отладка)
  default — все профили × все атаки × 1 сид × все 36 вопросов × 6 iter SECRET (тюнинг)
  final   — `--seeds 41,42,43` явно (3 сида, mean ± std для главы 4)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from attackrag.paths import (
    default_golden_qa_path,
    default_index_dir,
    default_poisoned_index_dir,
    repo_root,
)


def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _read_json(p: Path) -> dict[str, Any]:
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _stdout_summary_json(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "asr" in d and "out" in d:
            return d
    return None


def _git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root()),
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return ""


def _env_safe_snapshot() -> dict[str, str | None]:
    """ENV-снэпшот без секретов (значения API-ключей и т.п. не пишем)."""
    safe_keys = [
        "LLM_PROVIDER",
        "LLM_MODEL",
        "OLLAMA_MODEL",
        "OLLAMA_HOST",
        "RAG_TOP_K",
        "RAG_CANDIDATE_K",
        "RAG_RERANKER_ENABLED",
        "RAG_RERANKER_MODEL",
        "QDRANT_URL",
        "LLM_VERIFIER_PROVIDER",
        "LLM_VERIFIER_MODEL",
        "LLM_SEGMENTER_PROVIDER",
        "LLM_SEGMENTER_MODEL",
        "ATTACK_JUDGE_PROVIDER",
        "ATTACK_JUDGE_MODEL",
        "LLM_CACHE_DIR",
        "LLM_CACHE_GENERATOR",
    ]
    return {k: os.environ.get(k) for k in safe_keys}


def _aggregate_mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(max(var, 0.0))


def _fmt(v: Any) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Оркестратор экспериментов AttackRAG. Запускает профили × атаки × сиды через "
            "attackrag.cli.run_attacks как подпроцессы; собирает сводку с mean ± std."
        ),
    )
    p.add_argument("--index", type=Path, default=default_index_dir())
    p.add_argument("--index-poisoned", type=Path, default=None, help="Для backdoor: отравленный индекс")
    p.add_argument("--golden", type=Path, default=default_golden_qa_path())
    p.add_argument("--out", type=Path, default=None, help="Каталог runs/experiment_<ts> (по умолчанию в runs/)")
    p.add_argument(
        "--profiles",
        type=str,
        default="none,basic-filters,ragfort,hard",
        help="Через запятую: none | basic-filters | ragfort | hard (по умолчанию все 4).",
    )
    p.add_argument(
        "--attacks",
        type=str,
        default="pi,backdoor,secret",
        help="Через запятую: pi | backdoor | secret (по умолчанию все 3).",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Сиды через запятую. Дефолт: 42 (один сид). Финальный прогон: 41,42,43.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Все профили × все атаки × 1 сид × 5 вопросов × 1 iter SECRET (≤ 5 минут).",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="trials=3, secret-iters=3, secret-clusters=10 (≤ 25 минут).",
    )
    p.add_argument(
        "--backdoor-runtime-inject",
        action="store_true",
        help="Backdoor через runtime-инъекцию контекста (без отдельного индекса). Не для финальных цифр.",
    )
    p.add_argument(
        "--secret-lite",
        action="store_true",
        help="Упрощённый SECRET (без LLM-оптимизатора). Имеет смысл вместе с --smoke.",
    )
    p.add_argument(
        "--attack-leak-target",
        choices=["s-priv", "ground-truth"],
        default="s-priv",
        help="Критерий успеха атаки (по умолчанию s-priv).",
    )
    p.add_argument(
        "--s-priv-config",
        type=Path,
        default=None,
        help="YAML с literals/regexes для S_priv. Иначе data/s_priv.yaml при наличии.",
    )
    p.add_argument(
        "--limit-questions",
        type=int,
        default=None,
        help="Принудительно урезать golden до первых N вопросов (для smoke / fast). По умолчанию — выбирается режимом.",
    )
    p.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Количество испытаний на каждый PI/backdoor-вопрос. Если не задано — выбирается режимом (smoke=1, fast=3, default=5).",
    )
    p.add_argument(
        "--secret-iters",
        type=int,
        default=None,
        help="Итераций оптимизатора SECRET-атаки на один вопрос. Чем больше, тем сильнее атака; > 6 редко даёт прирост.",
    )
    p.add_argument(
        "--secret-clusters",
        type=int,
        default=None,
        help="Число кластеров k-means для SECRET (по корпусу). 25 — план; 8–10 для бедного корпуса; 50+ редко окупается.",
    )
    args = p.parse_args()

    root = repo_root()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out: Path = args.out or (root / "runs" / f"experiment_{ts}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    seeds = [int(s) for s in _parse_csv(args.seeds)]
    profiles = _parse_csv(args.profiles)
    attacks = _parse_csv(args.attacks)

    if args.smoke:
        # Все профили × все атаки, но минимальный объём данных и итераций. Цель —
        # проверить UI/CLI/подключения, а не получить осмысленные цифры.
        if len(seeds) > 1:
            print(f"[smoke] игнорирую --seeds {args.seeds}, использую только {seeds[0]}", file=sys.stderr)
            seeds = [seeds[0]]
        trials, secret_iters, secret_clusters = 1, 1, 4
        question_limit = args.limit_questions or 5
        mode_label = "smoke"
    elif args.fast:
        trials, secret_iters, secret_clusters = 3, 3, 10
        question_limit = args.limit_questions or 10
        mode_label = "fast"
    else:
        trials, secret_iters, secret_clusters = 5, 6, 25
        question_limit = args.limit_questions  # None = все вопросы
        mode_label = "default"

    # Явные --trials / --secret-iters / --secret-clusters перекрывают режим
    # (UI пробрасывает их при advanced-настройке, чтобы не зависеть от smoke/fast).
    if args.trials is not None:
        trials = args.trials
    if args.secret_iters is not None:
        secret_iters = args.secret_iters
    if args.secret_clusters is not None:
        secret_clusters = args.secret_clusters

    poisoned = args.index_poisoned or default_poisoned_index_dir()

    # Если запрошена backdoor и нет poisoned-индекса — собираем его один раз.
    needs_poison = ("backdoor" in attacks) and not args.backdoor_runtime_inject
    poison_built = False
    if needs_poison and not Path(poisoned).exists():
        print(f"[orchestrator] poisoned индекс {poisoned} отсутствует — собираю автоматически", file=sys.stderr)
        cmd_build = [
            sys.executable,
            "-m",
            "attackrag.cli.run_attacks",
            "--only-build-poison-index",
            "--out-poison-index",
            str(poisoned),
            "--backend",
            "qdrant",
        ]
        rb = subprocess.run(cmd_build, cwd=str(root), capture_output=True, text=True)
        (out / "logs" / "0_build_poison_index.log").write_text(
            f"cmd={' '.join(cmd_build)}\nrc={rb.returncode}\n\n--- stdout ---\n{rb.stdout}\n\n--- stderr ---\n{rb.stderr}\n",
            encoding="utf-8",
        )
        if rb.returncode != 0:
            raise SystemExit(
                f"Не удалось собрать poisoned индекс автоматически (см. logs/0_build_poison_index.log)."
            )
        poison_built = True

    # При limit_questions готовим временный golden, чтобы не править оригинал.
    actual_golden: Path = args.golden
    if question_limit is not None:
        try:
            full = json.loads(args.golden.read_text(encoding="utf-8"))
            if isinstance(full, list):
                trimmed = full[:question_limit]
                tmp = out / "_golden_subset.json"
                tmp.write_text(json.dumps(trimmed, ensure_ascii=False, indent=2), encoding="utf-8")
                actual_golden = tmp
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    # Запоминаем эффективную конфигурацию.
    config_doc: dict[str, Any] = {
        "mode": mode_label,
        "profiles": profiles,
        "attacks": attacks,
        "seeds": seeds,
        "trials_per_pi_question": trials,
        "secret_iters": secret_iters,
        "secret_clusters": secret_clusters,
        "question_limit": question_limit,
        "attack_leak_target": args.attack_leak_target,
        "backdoor_runtime_inject": args.backdoor_runtime_inject,
        "secret_lite": args.secret_lite,
        "index": str(args.index.resolve()),
        "index_poisoned": str(Path(poisoned).resolve()),
        "golden": str(actual_golden.resolve()),
    }
    (out / "config.yaml").write_text(
        # Простой YAML без зависимости от pyyaml ради чтения JSON-валидным:
        # для read-only артефакта это корректно, parsers YAML принимают JSON.
        json.dumps(config_doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": _git_commit(),
                "python": sys.version,
                "env_safe": _env_safe_snapshot(),
                "poisoned_index_built_now": poison_built,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    max_rc = 0

    for seed in seeds:
        for profile in profiles:
            for attack in attacks:
                # Подкаталог под этот прогон.
                run_dir = out / f"seed_{seed}" / f"profile_{profile}" / f"attack_{attack}"
                run_dir.mkdir(parents=True, exist_ok=True)
                rpath = run_dir / "report.json"

                # Бэкдор без runtime-inject требует существующий poisoned-индекс.
                if attack == "backdoor" and not args.backdoor_runtime_inject and not Path(poisoned).exists():
                    rows.append({
                        "seed": seed, "profile": profile, "attack": attack,
                        "returncode": 127, "report_path": "",
                        "asr": "", "fpr": "", "bpd": "",
                        "latency_overhead_ms": "", "kappa": "", "auc_ird": "",
                        "error": f"missing poisoned index: {poisoned}",
                    })
                    max_rc = max(max_rc, 1)
                    continue

                run_py: list[str] = [
                    sys.executable, "-m", "attackrag.cli.run_attacks",
                    "--attack", attack,
                    "--index", str(args.index),
                    "--golden", str(actual_golden),
                    "--out", str(rpath),
                    "--trials", str(trials),
                    "--seed", str(seed),
                    "--defense-profile", profile,
                    "--attack-leak-target", args.attack_leak_target,
                ]
                if attack == "backdoor" and not args.backdoor_runtime_inject:
                    run_py.extend(["--poisoned-index", str(poisoned)])
                if args.backdoor_runtime_inject and attack == "backdoor":
                    run_py.append("--backdoor-runtime-inject")
                if attack == "secret":
                    run_py.extend([
                        "--secret-iters", str(secret_iters),
                        "--secret-clusters", str(secret_clusters),
                    ])
                    if args.smoke or args.secret_lite:
                        run_py.append("--secret-lite")
                if args.s_priv_config is not None:
                    run_py.extend(["--s-priv-config", str(args.s_priv_config)])

                log_path = out / "logs" / f"s{seed}__{profile}__{attack}.log"
                r = subprocess.run(run_py, cwd=str(root), capture_output=True, text=True)
                log_path.write_text(
                    f"cmd={' '.join(run_py)!r}\nreturncode={r.returncode}\n\n"
                    f"--- stdout ---\n{r.stdout}\n\n--- stderr ---\n{r.stderr}\n",
                    encoding="utf-8",
                )
                max_rc = max(max_rc, r.returncode)

                rep = _read_json(rpath) if rpath.is_file() else {}
                m: dict[str, Any] = rep.get("metrics", {}) or {}
                err = ""
                if r.returncode != 0 and not m:
                    err = (r.stderr or r.stdout)[:1500]

                rows.append({
                    "seed": seed,
                    "profile": profile,
                    "attack": attack,
                    "returncode": r.returncode,
                    "report_path": str(rpath) if rpath.is_file() else "",
                    "asr": _fmt(m.get("asr")),
                    "fpr": _fmt(m.get("fpr")),
                    "fpr_legit_questions_only": _fmt(m.get("fpr_legit_questions_only")),
                    "bpd": _fmt(m.get("bpd")),
                    "bpd_legit_questions_only": _fmt(m.get("bpd_legit_questions_only")),
                    "latency_overhead_ms": _fmt(m.get("latency_overhead_ms")),
                    "kappa": _fmt(m.get("kappa")),
                    "auc_ird": _fmt(m.get("auc_ird")),
                    "stage_pass_rates": json.dumps(m.get("stage_pass_rates") or {}, ensure_ascii=False),
                    "error": err,
                })

    # Агрегируем по seed-ам внутри (profile × attack).
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["profile"], row["attack"])
        bucket = agg.setdefault(key, {
            "profile": row["profile"], "attack": row["attack"],
            "n_seeds_ok": 0, "n_seeds_fail": 0,
            "asr": [], "fpr": [], "fpr_legit_questions_only": [],
            "bpd": [], "bpd_legit_questions_only": [],
            "latency_overhead_ms": [], "kappa": [], "auc_ird": [],
        })
        if row["returncode"] != 0:
            bucket["n_seeds_fail"] += 1
            continue
        bucket["n_seeds_ok"] += 1
        for k in ("asr", "fpr", "fpr_legit_questions_only", "bpd",
                  "bpd_legit_questions_only", "latency_overhead_ms",
                  "kappa", "auc_ird"):
            v = row.get(k)
            try:
                if v not in (None, ""):
                    bucket[k].append(float(v))
            except (TypeError, ValueError):
                pass

    summary_rows: list[dict[str, Any]] = []
    for (prof, atk), b in agg.items():
        out_row: dict[str, Any] = {
            "profile": prof,
            "attack": atk,
            "n_seeds_ok": b["n_seeds_ok"],
            "n_seeds_fail": b["n_seeds_fail"],
        }
        for k in ("asr", "fpr", "fpr_legit_questions_only", "bpd",
                  "bpd_legit_questions_only", "latency_overhead_ms",
                  "kappa", "auc_ird"):
            mu, sd = _aggregate_mean_std(b[k])
            out_row[f"{k}_mean"] = round(mu, 6)
            out_row[f"{k}_std"] = round(sd, 6)
        summary_rows.append(out_row)

    summary_doc: dict[str, Any] = {
        "orchestrator": "run_experiment",
        "out_dir": str(out),
        "mode": mode_label,
        "profiles": profiles,
        "attacks": attacks,
        "seeds": seeds,
        "trials_per_pi_question": trials,
        "secret_iters": secret_iters,
        "question_limit": question_limit,
        "attack_leak_target": args.attack_leak_target,
        "rows": rows,             # per-(seed × profile × attack) исходные числа
        "aggregated": summary_rows,  # mean ± std по seed-ам
        "status": "ok" if max_rc == 0 else "error",
    }
    (out / "summary.json").write_text(
        json.dumps(summary_doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # CSV для главы 4: агрегированные строки (mean / std).
    csv_fields = [
        "profile", "attack", "n_seeds_ok", "n_seeds_fail",
        "asr_mean", "asr_std",
        "fpr_legit_questions_only_mean", "fpr_legit_questions_only_std",
        "bpd_legit_questions_only_mean", "bpd_legit_questions_only_std",
        "latency_overhead_ms_mean", "latency_overhead_ms_std",
        "kappa_mean", "kappa_std",
        "auc_ird_mean", "auc_ird_std",
    ]
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r2 in summary_rows:
            w.writerow({k: r2.get(k, "") for k in csv_fields})

    # Старый плоский reports/ для обратной совместимости со старыми скриптами.
    legacy_reports = out / "reports"
    legacy_reports.mkdir(exist_ok=True)
    for row in rows:
        if row["report_path"]:
            src = Path(row["report_path"])
            if src.is_file():
                dst = legacy_reports / f"{row['profile']}__{row['attack']}__s{row['seed']}.json"
                try:
                    shutil.copy2(src, dst)
                except OSError:
                    pass

    print(json.dumps({
        "orchestrator": "run_experiment",
        "out_dir": str(out),
        "mode": mode_label,
        "status": summary_doc["status"],
        "n_runs": len(rows),
        "n_aggregated": len(summary_rows),
    }, ensure_ascii=False, indent=2))

    if max_rc != 0:
        raise SystemExit(max_rc)


if __name__ == "__main__":
    main()
