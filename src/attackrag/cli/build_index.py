from __future__ import annotations

import argparse
from pathlib import Path

from attackrag.chunking import chunk_documents
from attackrag.cli.corpus.fetch_wikipedia import stream_wikipedia
from attackrag.documents import load_markdown_corpus
from attackrag.embeddings import EmbeddingModel
from attackrag.paths import default_corpus_dir, default_index_dir, repo_root
from attackrag.vector_stores.loader import BACKENDS, build_vector_store


def _ensure_wikipedia_corpus(
    *,
    wiki_dir: Path,
    required_docs: int,
    wiki_config: str,
    min_chars: int,
    max_chars_per_doc: int,
) -> int:
    wiki_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(wiki_dir.glob("*.md")))
    if existing >= required_docs:
        return existing
    need = required_docs - existing
    written = stream_wikipedia(
        config=wiki_config,
        max_docs=need,
        out_dir=wiki_dir,
        min_chars=min_chars,
        max_chars_per_doc=max_chars_per_doc,
        skip_first=existing,
    )
    return existing + written


def _load_merged_docs(base_corpus: Path, wiki_corpus: Path):
    docs = load_markdown_corpus(base_corpus) + load_markdown_corpus(wiki_corpus)
    # На случай коллизий doc_id между разными директориями.
    seen: set[str] = set()
    for i, d in enumerate(docs):
        if d.doc_id in seen:
            docs[i] = type(d)(doc_id=f"{d.path.parent.name}__{d.doc_id}", path=d.path, text=d.text)
        seen.add(docs[i].doc_id)
    return docs


def main() -> None:
    p = argparse.ArgumentParser(description="Построить векторный индекс по корпусу .md")
    p.add_argument(
        "--corpus",
        type=Path,
        default=default_corpus_dir(),
        help="Каталог с Markdown-документами",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=default_index_dir(),
        help="Каталог для сохранения индекса",
    )
    p.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    p.add_argument(
        "--backend",
        choices=list(BACKENDS),
        default="qdrant",
        help="По умолчанию qdrant (см. docker compose / embedded). numpy/faiss/chroma — legacy.",
    )
    p.add_argument(
        "--skip-wiki",
        action="store_true",
        help="Не подмешивать Wikipedia: индексируется только --corpus (быстрый dev/smoke).",
    )
    p.add_argument("--max-chars", type=int, default=900)
    p.add_argument("--overlap", type=int, default=120)
    p.add_argument(
        "--wiki-corpus",
        type=Path,
        default=repo_root() / "data" / "corpus_wikipedia",
        help="Каталог markdown-статей Википедии",
    )
    p.add_argument(
        "--wiki-required-docs",
        type=int,
        default=300,
        help="Минимум статей Wikipedia (игнорируется с --skip-wiki). Рекомендация по диссертации: 200–500.",
    )
    p.add_argument(
        "--wiki-config",
        default="20231101.ru",
        help="Конфиг HF wikimedia/wikipedia, например 20231101.ru или 20231101.en",
    )
    p.add_argument("--wiki-min-chars", type=int, default=400)
    p.add_argument("--wiki-max-chars-per-doc", type=int, default=14_000)
    args = p.parse_args()

    if args.skip_wiki:
        wiki_count = 0
        docs = load_markdown_corpus(args.corpus)
    else:
        wiki_count = _ensure_wikipedia_corpus(
            wiki_dir=args.wiki_corpus,
            required_docs=args.wiki_required_docs,
            wiki_config=args.wiki_config,
            min_chars=args.wiki_min_chars,
            max_chars_per_doc=args.wiki_max_chars_per_doc,
        )
        if wiki_count < args.wiki_required_docs:
            raise RuntimeError(
                f"Недостаточно Wikipedia-статей: {wiki_count}/{args.wiki_required_docs}. "
                "Проверьте интернет/HF доступ, уменьшите --wiki-required-docs или используйте --skip-wiki."
            )
        docs = _load_merged_docs(args.corpus, args.wiki_corpus)

    chunks = chunk_documents(docs, max_chars=args.max_chars, overlap=args.overlap)
    embedder = EmbeddingModel(args.embedding_model)
    build_vector_store(args.out, chunks, embedder, backend=args.backend)
    print(
        f"Индекс сохранён: {args.out} "
        f"(backend={args.backend}, docs={len(docs)}, wiki_docs={wiki_count}, чанков={len(chunks)})"
    )


if __name__ == "__main__":
    main()
