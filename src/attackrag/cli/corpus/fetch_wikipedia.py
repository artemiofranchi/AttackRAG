from __future__ import annotations

import argparse
import re
from pathlib import Path

from datasets import load_dataset


def _safe_filename(title: str, idx: int) -> str:
    base = (title or "").strip()[:100]
    for ch in '<>:"/\\|?*\n\r\t':
        base = base.replace(ch, "_")
    base = re.sub(r"\s+", "_", base).strip("._") or "article"
    return f"{idx:06d}_{base}.md"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n\n[… обрезано …]"


def stream_wikipedia(
    *,
    config: str,
    max_docs: int,
    out_dir: Path,
    min_chars: int,
    max_chars_per_doc: int,
    skip_first: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "wikimedia/wikipedia",
        config,
        split="train",
        streaming=True,
    )
    it = iter(ds)
    for _ in range(skip_first):
        try:
            next(it)
        except StopIteration:
            return 0

    written = 0
    idx = skip_first
    while written < max_docs:
        try:
            row = next(it)
        except StopIteration:
            break
        idx += 1
        title = str(row.get("title", ""))
        text = str(row.get("text", ""))
        url = str(row.get("url", ""))
        if len(text) < min_chars:
            continue
        body = _truncate(text, max_chars_per_doc)
        content = f"# {title}\n\nИсточник: {url}\n\n{body}\n"
        path = out_dir / _safe_filename(title, idx)
        path.write_text(content, encoding="utf-8")
        written += 1
    return written


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Скачать статьи Википедии через Hugging Face (wikimedia/wikipedia, streaming). "
            "Это предпочтительнее произвольного парсинга HTML: лицензия CC BY-SA, стабильная схема."
        )
    )
    p.add_argument(
        "--config",
        default="20231101.ru",
        help="Имя конфига датасета, например 20231101.ru (русская Вики) или 20231101.en",
    )
    p.add_argument("--max-docs", type=int, default=500, help="Сколько статей сохранить")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/corpus_wikipedia"),
        help="Каталог для .md файлов",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=400,
        help="Пропускать слишком короткие статьи",
    )
    p.add_argument(
        "--max-chars-per-doc",
        type=int,
        default=14_000,
        help="Обрезать длинные статьи (символов текста статьи)",
    )
    p.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="Пропустить первые N записей потока (для другой выборки без скачивания всего с начала)",
    )
    args = p.parse_args()

    n = stream_wikipedia(
        config=args.config,
        max_docs=args.max_docs,
        out_dir=args.out,
        min_chars=args.min_chars,
        max_chars_per_doc=args.max_chars_per_doc,
        skip_first=args.skip_first,
    )
    print(f"Сохранено статей: {n} -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
