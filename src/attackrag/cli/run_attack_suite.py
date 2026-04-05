from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from attackrag.paths import default_golden_qa_path, default_index_dir, repo_root


def _run(cmd: list[str]) -> dict:
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    if not lines:
        return {"stdout": p.stdout}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"stdout": p.stdout}


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогнать PI/backdoor/SECRET + набор защит")
    parser.add_argument("--index", type=Path, default=default_index_dir())
    parser.add_argument("--golden", type=Path, default=default_golden_qa_path())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--secret-iters", type=int, default=10)
    parser.add_argument("--backdoor-runtime-inject", action="store_true")
    parser.add_argument("--poisoned-index", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="JSON summary file")
    parser.add_argument("--leak-mode", choices=["exact", "regex", "judge"], default="exact")
    parser.add_argument(
        "--defense-profiles",
        type=str,
        default="none,input,verifier,all,leaksealer,controlnet,ragfort",
        help="Комма-разделитель: none,input,verifier,all,leaksealer,controlnet,ragfort",
    )
    parser.add_argument(
        "--guard-context-block",
        type=str,
        default="секрет,пароль,конфиденциаль,триггер",
        help="Подстроки (через запятую) для DataFilter (guard-context-block)",
    )
    args = parser.parse_args()

    profiles = [p.strip() for p in args.defense_profiles.split(",") if p.strip()]
    base = [
        sys.executable,
        "-m",
        "attackrag.cli.run_attacks",
        "--golden",
        str(args.golden),
        "--seed",
        str(args.seed),
        "--leak-mode",
        args.leak_mode,
    ]

    def _flags_for_profile(profile: str) -> list[str]:
        if profile == "none":
            return []
        if profile == "input":
            return ["--guard-input-llm"]
        if profile == "verifier":
            return ["--guard-output-llm"]
        if profile == "all":
            return [
                "--guard-input-llm",
                "--guard-output-llm",
                "--guard-context-block",
                args.guard_context_block,
            ]
        if profile == "leaksealer":
            return ["--enable-leaksealer"]
        if profile == "controlnet":
            return ["--enable-controlnet"]
        if profile == "ragfort":
            return ["--enable-ragfort", "--guard-output-llm", "--guard-context-block", args.guard_context_block]
        raise ValueError(f"unknown defense profile: {profile}")

    out: dict[str, dict[str, dict]] = {}
    attacks = ["pi", "backdoor", "secret"]
    for attack in attacks:
        out[attack] = {}
        for prof in profiles:
            extra_flags = _flags_for_profile(prof)
            if attack == "pi":
                cmd = base + ["--attack", "pi", "--index", str(args.index), "--trials", str(args.trials)] + extra_flags
            elif attack == "secret":
                cmd = (
                    base
                    + [
                        "--attack",
                        "secret",
                        "--index",
                        str(args.index),
                        "--secret-iters",
                        str(args.secret_iters),
                    ]
                    + extra_flags
                )
            else:
                # backdoor
                if args.backdoor_runtime_inject:
                    cmd = (
                        base
                        + ["--attack", "backdoor", "--index", str(args.index), "--backdoor-runtime-inject", "--trials", "1"]
                        + extra_flags
                    )
                else:
                    if args.poisoned_index is None:
                        raise SystemExit("Для backdoor без runtime-inject укажите --poisoned-index")
                    cmd = base + ["--attack", "backdoor", "--poisoned-index", str(args.poisoned_index), "--trials", "1"] + extra_flags
            out[attack][prof] = _run(cmd)

    if args.out is None:
        (repo_root() / "runs").mkdir(parents=True, exist_ok=True)
        args.out = repo_root() / "runs" / "attack_suite_latest.json"
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": out}, ensure_ascii=False))


if __name__ == "__main__":
    main()

