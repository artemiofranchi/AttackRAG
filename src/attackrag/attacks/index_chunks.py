from __future__ import annotations

import json
from pathlib import Path

from attackrag.chunking import Chunk


def load_chunks_json(index_dir: Path) -> list[Chunk]:
    p = index_dir / "chunks.json"
    if not p.is_file():
        raise FileNotFoundError(f"Нет {p}: индекс без chunks.json (пересоберите индекс)")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in raw]
