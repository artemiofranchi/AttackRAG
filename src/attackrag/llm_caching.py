from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from attackrag.llm import LLMClient
from attackrag.paths import repo_root


def _cache_dir() -> Path:
    d = os.environ.get("LLM_CACHE_DIR", "").strip()
    base = Path(d) if d else (repo_root() / "runs" / "llm_cache")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _key(
    *,
    role: str,
    model_hint: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    raw = f"{role}\n{model_hint}\n{max_tokens}\n{temperature}\n{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CachedLLM(LLMClient):
    """On-disk кэш вызовов `complete` (NFR-3). Ключ: SHA256 от роли, модели, параметров и промпта."""

    def __init__(self, inner: LLMClient, *, role: str = "GENERATOR", model_hint: str = "") -> None:
        self._inner = inner
        self._role = role
        self._model_hint = model_hint or type(inner).__name__

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        h = _key(
            role=self._role,
            model_hint=self._model_hint,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        path = _cache_dir() / f"{h}.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return str(data.get("text", ""))
        text = self._inner.complete(prompt, max_tokens=max_tokens, temperature=temperature)
        path.write_text(
            json.dumps({"text": text, "role": self._role, "model_hint": self._model_hint}, ensure_ascii=False),
            encoding="utf-8",
        )
        return text
