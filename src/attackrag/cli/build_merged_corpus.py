from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from attackrag.paths import default_corpus_dir, default_corpus_merged_dir, repo_root


def _gather_md_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.glob("**/*.md"))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Собрать data/corpus_merged: копия базового корпуса + опционально Wikipedia."
    )
    p.add_argument("--base", type=Path, default=default_corpus_dir(), help="Базовый корпус (обязателен).")
    p.add_argument("--wiki", type=Path, default=repo_root() / "data" / "corpus_wikipedia", help="Wikipedia .md")
    p.add_argument("--extra", type=Path, nargs="*", default=[], help="Дополнительные корни с .md")
    p.add_argument("--out", type=Path, default=default_corpus_merged_dir())
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for root in (args.base, args.wiki, *args.extra):
        files.extend(_gather_md_files(root))

    for i, f in enumerate(files):
        name = f"{i:05d}_{f.stem}.md" if f.suffix else f"{i:05d}.md"
        shutil.copy2(f, args.out / name)

    print(f"Merged corpus: {args.out} ({len(files)} .md files)")


if __name__ == "__main__":
    main()
