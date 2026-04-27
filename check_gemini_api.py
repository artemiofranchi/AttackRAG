#!/usr/bin/env python3
"""Быстрая проверка доступа к Gemini через OpenAI-compatible API (как VERIFIER в llm_roles)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("Нет GEMINI_API_KEY / GOOGLE_API_KEY в окружении или .env", file=sys.stderr)
        return 2

    base = (
        os.environ.get("RAGAS_GEMINI_OPENAI_BASE_URL")
        or os.environ.get("GEMINI_OPENAI_BASE_URL")
        or "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    model = os.environ.get("RAGAS_GEMINI_MODEL") or os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash"
    timeout = float(os.environ.get("GEMINI_SMOKE_TIMEOUT", "60"))

    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=base.rstrip("/") + "/", timeout=timeout)
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": 'Ответь одним словом: "ok".'}],
            max_tokens=16,
            temperature=0.0,
        )
        text = (r.choices[0].message.content or "").strip()
        print(f"OK model={model!r} base={base!r}\nОтвет: {text!r}")
        return 0
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        print(f"  model={model!r} base={base!r} timeout={timeout}s", file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
