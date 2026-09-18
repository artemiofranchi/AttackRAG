"""DEPRECATED. Используйте `attack-rag-experiment` (run_experiment.py).

Этот старый CLI оперировал семантикой профилей `none/input/verifier/all/ragfort`,
которая отличается от зафиксированной в плане (`none/basic-filters/ragfort/hard`)
и от формализации (§3, `tab:traceability`). Сохраняем модуль в репозитории как
тонкий stub: при вызове печатает понятную ошибку и подсказывает миграцию.
Полностью удалять не стали ради старых скриптов; точку входа из pyproject убрали.
"""

from __future__ import annotations

import sys


_DEPRECATION_MESSAGE = (
    "run_attack_suite (attack-rag-run-attack-suite) больше не поддерживается.\n"
    "Используй универсальный оркестратор:\n\n"
    "    python -m attackrag.cli.run_experiment "
    "--profiles none,basic-filters,ragfort,hard --attacks pi,backdoor,secret\n\n"
    "Или режимы:\n"
    "    --smoke   (≤ 5 минут, минимум данных)\n"
    "    --fast    (≤ 25 минут, отладка)\n"
    "    --seeds 41,42,43  (финальный прогон с mean ± std)\n"
)


def main() -> None:
    print(_DEPRECATION_MESSAGE, file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
