from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from attackrag.paths import default_golden_qa_path, default_index_dir, default_poisoned_index_dir, repo_root


def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _read_run_json(p: Path) -> dict[str, Any]:
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Оркестратор: циклы по профилям/атакам, подпроцессы attack-rag-run-attacks, summary.json/csv.",
    )
    p.add_argument("--index", type=Path, default=default_index_dir())
    p.add_argument("--index-poisoned", type=Path, default=None, help="Для backdoor: отравленный индекс")
    p.add_argument("--golden", type=Path, default=default_golden_qa_path())
    p.add_argument("--out", type=Path, default=None, help="Каталог runs/experiment_<ts> (по умолчанию в runs/)")
    p.add_argument(
        "--profiles",
        type=str,
        default="none",
        help="Через запятую: none | basic-filters | ragfort | hard",
    )
    p.add_argument(
        "--attacks",
        type=str,
        default="pi",
        help="Через запятую: pi | backdoor | secret",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Сиды через запятую (каждое сочетание profile×attack×seed = один прогон).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="1 профиль, 1 атака, 1 сид, trials=1, secret-iters=1, secret-lite",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="trials=3, secret-iters=3, secret-clusters=4 (сокращение времени).",
    )
    p.add_argument(
        "--backdoor-runtime-inject",
        action="store_true",
        help="Проброс в run_attacks: runtime backdoor без отдельного индекса",
    )
    p.add_argument(
        "--secret-lite",
        action="store_true",
        help="Проброс: упрощённый SECRET (можно вместе с --smoke).",
    )
    p.add_argument(
        "--attack-leak-target",
        choices=["s-priv", "ground-truth"],
        default="s-priv",
        help="Проброс в run_attacks: критерий успеха атаки (по умолчанию S_priv).",
    )
    p.add_argument(
        "--s-priv-config",
        type=Path,
        default=None,
        help="Проброс: YAML с literals/regexes для S_priv (иначе data/s_priv.yaml при наличии).",
    )
    args = p.parse_args()

    root = repo_root()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out: Path = args.out or (root / "runs" / f"experiment_{ts}")
    out.mkdir(parents=True, exist_ok=True)
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    seeds = [int(s) for s in _parse_csv(args.seeds)]
    if args.smoke:
        seeds = [seeds[0]] if seeds else [42]
        profiles = ["none"]
        attacks = ["pi"]
        trials, secret_iters, secret_clusters = 1, 1, 4
    else:
        profiles = _parse_csv(args.profiles)
        attacks = _parse_csv(args.attacks)
        if args.fast:
            trials, secret_iters, secret_clusters = 3, 3, 4
        else:
            trials, secret_iters, secret_clusters = 5, 15, 8

    poisoned = args.index_poisoned
    if poisoned is None:
        poisoned = default_poisoned_index_dir()

    rows: list[dict[str, Any]] = []
    max_rc = 0
    for seed in seeds:
        for profile in profiles:
            for attack in attacks:
                if attack == "backdoor" and not args.backdoor_runtime_inject:
                    if not Path(poisoned).exists():
                        rows.append(
                            {
                                "profile": profile,
                                "attack": attack,
                                "seed": seed,
                                "returncode": 127,
                                "report_path": "",
                                "asr": "",
                                "fpr": "",
                                "bpd": "",
                                "latency_overhead_ms": "",
                                "kappa": "",
                                "error": f"missing poisoned index: {poisoned}",
                            }
                        )
                        max_rc = max(max_rc, 1)
                        continue
                rname = f"{profile}__{attack}__s{seed}.json"
                rpath = reports / rname
                run_py: list[str] = [
                    sys.executable,
                    "-m",
                    "attackrag.cli.run_attacks",
                    "--attack",
                    attack,
                    "--index",
                    str(args.index),
                    "--golden",
                    str(args.golden),
                    "--out",
                    str(rpath),
                    "--trials",
                    str(trials),
                    "--seed",
                    str(seed),
                    "--defense-profile",
                    profile,
                ]
                if attack == "backdoor" and not args.backdoor_runtime_inject:
                    run_py.extend(["--poisoned-index", str(poisoned)])
                if args.backdoor_runtime_inject and attack == "backdoor":
                    run_py.append("--backdoor-runtime-inject")
                if attack == "secret":
                    run_py.extend(
                        [
                            "--secret-iters",
                            str(secret_iters),
                            "--secret-clusters",
                            str(secret_clusters),
                        ]
                    )
                    if args.smoke or args.secret_lite:
                        run_py.append("--secret-lite")
                run_py.extend(["--attack-leak-target", args.attack_leak_target])
                if args.s_priv_config is not None:
                    run_py.extend(["--s-priv-config", str(args.s_priv_config)])

                log_path = out / "logs" / f"{profile}__{attack}__s{seed}.log"
                r = subprocess.run(run_py, cwd=str(root), capture_output=True, text=True)
                log_path.write_text(
                    f"cmd={' '.join(run_py)!r}\nreturncode={r.returncode}\n\n--- stdout ---\n{r.stdout}\n\n--- stderr ---\n{r.stderr}\n",
                    encoding="utf-8",
                )
                max_rc = max(max_rc, r.returncode)
                js = _stdout_summary_json(r.stdout) or {}
                if r.returncode != 0 and not js:
                    err = (r.stderr or r.stdout)[:2000]
                else:
                    err = ""
                rep_data = _read_run_json(rpath) if rpath.is_file() else {}
                m = rep_data.get("metrics") or {}
                rows.append(
                    {
                        "profile": profile,
                        "attack": attack,
                        "seed": seed,
                        "returncode": r.returncode,
                        "report_path": str(rpath) if rpath.is_file() else "",
                        "asr": _fmt(js.get("asr", m.get("asr"))),
                        "fpr": _fmt(js.get("fpr", m.get("fpr"))),
                        "bpd": _fmt(js.get("bpd", m.get("bpd"))),
                        "latency_overhead_ms": _fmt(
                            js.get("latency_overhead_ms", m.get("latency_overhead_ms"))
                        ),
                        "kappa": _fmt(js.get("kappa", m.get("kappa"))),
                        "error": err,
                    }
                )

    summary: dict[str, Any] = {
        "orchestrator": "run_experiment",
        "out_dir": str(out),
        "profiles": profiles,
        "attacks": attacks,
        "seeds": seeds,
        "smoke": args.smoke,
        "fast": bool(args.fast) and not args.smoke,
        "trials": trials,
        "secret_iters": secret_iters,
        "secret_clusters": secret_clusters,
        "attack_leak_target": args.attack_leak_target,
        "rows": rows,
        "status": "ok" if max_rc == 0 else "error",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "profile",
                "attack",
                "seed",
                "returncode",
                "asr",
                "fpr",
                "bpd",
                "latency_overhead_ms",
                "kappa",
                "report_path",
                "error",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(
                {k: row.get(k, "") for k in w.fieldnames}  # type: ignore[assignment, arg-type]
            )

    print(
        json.dumps(
            {
                "orchestrator": summary.get("orchestrator"),
                "out_dir": summary.get("out_dir"),
                "status": summary.get("status"),
                "n_runs": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if max_rc != 0:
        raise SystemExit(max_rc)


def _fmt(v: Any) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


if __name__ == "__main__":
    main()
