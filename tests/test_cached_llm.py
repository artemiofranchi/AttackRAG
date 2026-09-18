"""Юнит-тесты CachedLLM (NFR-3): один и тот же промпт ⇒ один вызов inner LLM."""

from __future__ import annotations

from pathlib import Path

import pytest

from attackrag.llm import LLMClient
from attackrag.llm_caching import CachedLLM


class _CountingLLM(LLMClient):
    def __init__(self, prefix: str = "ans") -> None:
        self.calls = 0
        self._prefix = prefix

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        self.calls += 1
        return f"{self._prefix}:{prompt[:32]}"


def test_cache_hits_on_second_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    inner = _CountingLLM()
    cached = CachedLLM(inner, role="GENERATOR", model_hint="test-model")

    a = cached.complete("hello world", max_tokens=64, temperature=0.0)
    b = cached.complete("hello world", max_tokens=64, temperature=0.0)
    assert a == b
    assert inner.calls == 1, f"Cache miss on identical prompt: calls={inner.calls}"


def test_cache_miss_on_different_temperature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    inner = _CountingLLM()
    cached = CachedLLM(inner, role="GENERATOR", model_hint="test-model")

    cached.complete("p", max_tokens=64, temperature=0.0)
    cached.complete("p", max_tokens=64, temperature=0.1)
    assert inner.calls == 2


def test_cache_miss_on_different_role(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Кэш разделяется по ролям — VERIFIER и GENERATOR не пересекаются."""
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    inner = _CountingLLM()
    gen = CachedLLM(inner, role="GENERATOR", model_hint="m")
    ver = CachedLLM(inner, role="VERIFIER", model_hint="m")
    gen.complete("same prompt")
    ver.complete("same prompt")
    assert inner.calls == 2


def test_cache_independent_of_model_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Смена модели должна инвалидировать кэш — иначе результаты Ollama / Gemini пересекутся."""
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    inner = _CountingLLM()
    a = CachedLLM(inner, role="GENERATOR", model_hint="ollama:llama3.1:8b")
    b = CachedLLM(inner, role="GENERATOR", model_hint="gemini-2.0-flash")
    a.complete("prompt")
    b.complete("prompt")
    assert inner.calls == 2
