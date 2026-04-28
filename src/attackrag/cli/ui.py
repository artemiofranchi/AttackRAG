from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Точка входа `attack-rag-ui` → `streamlit run apps/streamlit_app.py`.

    По умолчанию отключаем file-watcher Streamlit
    (`STREAMLIT_SERVER_FILE_WATCHER_TYPE=none`). Иначе watcher на старте обходит
    дерево импортов нашего apps/streamlit_app.py — а там через цепочку
    `attackrag.* → sentence_transformers → transformers` подтягивается весь
    зоопарк image-моделей `transformers`, многие из которых требуют
    `torchvision` (его в проекте нет — мы работаем только с текстом). Получаем
    десятки `ModuleNotFoundError: torchvision` в логах, не влияющих на работу,
    но мешающих читать stdout. Watcher нам не нужен: UI запускают одной
    командой, не редактируя на лету.

    При желании включить watcher (например, для разработки самого UI)
    переопредели переменную окружения вручную:

        STREAMLIT_SERVER_FILE_WATCHER_TYPE=auto attack-rag-ui
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[3]  # src/attackrag/cli/ui.py
    app = repo_root / "apps" / "streamlit_app.py"
    if not app.is_file():
        raise SystemExit(f"Не найден {app}")
    env = {
        **os.environ,
        "PYTHONPATH": str(repo_root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    env.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
    # Наш UI рендерит свой собственный summary, шапка ассистента/usage-метрик
    # Streamlit в случае диплом-демо только мешает.
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=str(repo_root), env=env))


if __name__ == "__main__":
    main()
