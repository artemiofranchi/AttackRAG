from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Точка входа `attack-rag-ui` → `streamlit run apps/streamlit_app.py`."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]  # src/attackrag/cli/ui.py
    app = repo_root / "apps" / "streamlit_app.py"
    if not app.is_file():
        raise SystemExit(f"Не найден {app}")
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=str(repo_root), env=env))


if __name__ == "__main__":
    main()
