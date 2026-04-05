from __future__ import annotations

import os
from abc import ABC, abstractmethod

from openai import OpenAI


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str: ...


_SYSTEM_RU = (
    "Ты помощник. Отвечай кратко и по-русски. Используй только факты из предоставленного контекста; "
    "если данных нет — так и скажи."
)


class GeminiLLM(LLMClient):
    """Google Gemini через SDK `google-genai` (переменные GEMINI_API_KEY или GOOGLE_API_KEY)."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
    ) -> None:
        from google import genai

        key = api_key if api_key is not None else (
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        if not key:
            raise RuntimeError(
                "Для Gemini задайте GEMINI_API_KEY или GOOGLE_API_KEY (или api_key=...)."
            )
        self._client = genai.Client(api_key=key)
        self._model = model

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            system_instruction=_SYSTEM_RU,
        )
        r = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=cfg,
        )
        return _gemini_response_text(r).strip()


def _gemini_response_text(response: object) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    cands = getattr(response, "candidates", None) or []
    parts: list[str] = []
    for c in cands:
        content = getattr(c, "content", None)
        if content is None:
            continue
        for p in getattr(content, "parts", None) or []:
            t = getattr(p, "text", None)
            if t:
                parts.append(str(t))
    return "".join(parts)


class OpenAICompatLLM(LLMClient):
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "Не задан API-ключ: переменная окружения OPENAI_API_KEY "
                "(или передайте api_key=... в конструктор)."
            )
        self._client = OpenAI(api_key=key, base_url=base_url or os.environ.get("OPENAI_BASE_URL"))
        self._model = model

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        r = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_RU},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (r.choices[0].message.content or "").strip()


class OllamaLLM(LLMClient):
    """Локальные модели Ollama через OpenAI-совместимый API (`/v1/chat/completions`).

    Старый `POST /api/generate` на некоторых установках даёт 404 (другой сервис на порту или урезанный прокси).
    """

    def __init__(
        self,
        *,
        model: str,
        host: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        base = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        key = api_key if api_key is not None else os.environ.get("OLLAMA_API_KEY")
        self._client = OpenAI(
            base_url=f"{base}/v1",
            api_key=key or "ollama",
            timeout=120.0,
        )

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        r = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_RU},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (r.choices[0].message.content or "").strip()

    def close(self) -> None:
        self._client.close()
