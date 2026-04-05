from __future__ import annotations

import json
from pathlib import Path
from typing import Any

META_FILE = "meta.json"


def write_meta(directory: Path, payload: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / META_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_meta(directory: Path) -> dict[str, Any]:
    p = directory / META_FILE
    if not p.is_file():
        raise FileNotFoundError(f"Нет {p}: сначала attack-rag-build-index")
    return json.loads(p.read_text(encoding="utf-8"))
