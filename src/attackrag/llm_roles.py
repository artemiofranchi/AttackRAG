"""Фабрики LLMClient по ролям (см. tab:llm_roles в docs/thesis_formalization.tex)."""

from __future__ import annotations

import os

from attackrag.llm import LLMClient, OllamaLLM, OpenAICompatLLM


def _openai_compat_from_env(
    prefix: str,
    *,
    fallback_ragas_gemini: bool = True,
) -> OpenAICompatLLM | OllamaLLM | None:
    """LLM_{PREFIX}_* или для VERIFIER/SEGMENTER/JUDGE — RAGAS Gemini как fallback."""
    prov = (os.environ.get(f"LLM_{prefix}_PROVIDER") or "").strip().lower()
    if prov in {"", "default"} and fallback_ragas_gemini:
        if prefix in {"VERIFIER", "SEGMENTER", "JUDGE"} or prefix == "ATTACK_JUDGE":
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if key:
                base = (
                    os.environ.get(f"LLM_{prefix}_BASE_URL")
                    or os.environ.get("RAGAS_GEMINI_OPENAI_BASE_URL")
                    or os.environ.get("GEMINI_OPENAI_BASE_URL")
                    or "https://generativelanguage.googleapis.com/v1beta/openai/"
                )
                model = (
                    os.environ.get(f"LLM_{prefix}_MODEL")
                    or os.environ.get("RAGAS_GEMINI_MODEL")
                    or "gemini-2.0-flash"
                )
                return OpenAICompatLLM(model=model, base_url=base, api_key=key)
    if prov == "ollama":
        host = os.environ.get(f"LLM_{prefix}_OLLAMA_HOST") or os.environ.get("OLLAMA_HOST")
        model = os.environ.get(f"LLM_{prefix}_MODEL") or os.environ.get("OLLAMA_MODEL") or "llama3.1:8b"
        return OllamaLLM(model=model, host=host)
    if prov in {"openai_compat", "openai"}:
        model = os.environ.get(f"LLM_{prefix}_MODEL") or "gemini-2.0-flash"
        key = os.environ.get(f"LLM_{prefix}_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base = os.environ.get(f"LLM_{prefix}_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if not key:
            return None
        return OpenAICompatLLM(model=model, base_url=base, api_key=key)
    return None


def make_verifier_llm() -> LLMClient:
    llm = _openai_compat_from_env("VERIFIER", fallback_ragas_gemini=True)
    if llm is not None:
        return llm
    raise RuntimeError(
        "Задайте LLM_VERIFIER_* или GEMINI_API_KEY + RAGAS_GEMINI_MODEL для верификатора (Draft-then-Verify)."
    )


def make_segmenter_llm() -> LLMClient:
    llm = _openai_compat_from_env("SEGMENTER", fallback_ragas_gemini=True)
    if llm is not None:
        return llm
    # Для dev: тот же Ollama, что и GENERATOR
    ollama_model = os.environ.get("OLLAMA_MODEL") or "llama3.1:8b"
    return OllamaLLM(model=ollama_model, host=os.environ.get("OLLAMA_HOST"))


def make_judge_llm() -> LLMClient:
    """Судья для LeakDetector (JUDGE): env ATTACK_JUDGE / GEMINI, иначе как VERIFIER."""
    mode = (os.environ.get("ATTACK_JUDGE_PROVIDER") or "").strip().lower()
    if mode in ("ollama", "llama", "local"):
        m = (
            os.environ.get("ATTACK_JUDGE_MODEL")
            or os.environ.get("RAGAS_OLLAMA_MODEL")
            or os.environ.get("OLLAMA_MODEL")
            or "llama3.1"
        )
        host = os.environ.get("ATTACK_JUDGE_OLLAMA_HOST") or os.environ.get("OLLAMA_HOST")
        return OllamaLLM(model=m, host=host)
    if mode == "openai" and os.environ.get("OPENAI_API_KEY"):
        m = os.environ.get("ATTACK_JUDGE_MODEL") or "gpt-4o-mini"
        return OpenAICompatLLM(model=m, base_url=os.environ.get("OPENAI_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY"))
    if mode in {"gemini", ""} and (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        base = os.environ.get("RAGAS_GEMINI_OPENAI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/"
        m = os.environ.get("ATTACK_JUDGE_MODEL") or "gemini-2.0-flash"
        return OpenAICompatLLM(model=m, base_url=base, api_key=key)  # type: ignore[arg-type]
    return make_verifier_llm()
