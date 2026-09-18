"""Тесты соблюдения чёрно-ящичного threat model и контракта SECRET-атаки.

Проверяем:
  * `default_poison_markdown` НЕ содержит секретное значение (Step 5: атака
    инжектирует только инструкции, retrieval подтаскивает реальный секрет
    из легитимного `internal_security_memo`-чанка).
  * `find_target_cluster` сначала ищет уникальные алфанумерические токены
    из ground_truth (NBX-STG-, whsec_…), потом — семантическое сходство.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attackrag.attacks.corpus_poison import default_poison_markdown
from attackrag.attacks.secret_lite import find_target_cluster


def test_default_poison_markdown_does_not_embed_secret() -> None:
    secret = "NBX-STG-7f3c91e4-ABCDEF"
    md = default_poison_markdown(trigger="agrowatch staging tariff", secret_value=secret)
    assert secret not in md, "Poison-чанк не должен содержать реальное значение секрета"
    lower = md.lower()
    # Тематические триггеры остаются — без них поисковик не вытащит инструкцию.
    assert any(k in lower for k in ("staging", "api", "тариф")), (
        "Poison-чанк должен содержать тематические триггеры для retrieval"
    )


@dataclass
class _Chunk:
    """Минимальный stub Chunk: для find_target_cluster нужно только `.text`."""

    text: str


class _DummyEmb:
    model_name = "test/tiny"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 8), dtype=np.float32)
        for i, t in enumerate(texts):
            h = abs(hash(t)) % (2**31)
            rng = np.random.default_rng(h)
            out[i] = rng.random(8, dtype=np.float32)
        return out


def test_find_target_cluster_prefers_unique_token_substring() -> None:
    """Если в одном чанке есть точная подстрока NBX-STG-... из ground_truth — берём именно его кластер."""
    chunks = [
        _Chunk("обычный текст про животных"),
        _Chunk("technical memo: NBX-STG-7f3c91e4 confidential"),
        _Chunk("random political news"),
    ]
    labels = np.array([0, 1, 2])
    chunk_emb = _DummyEmb().encode([c.text for c in chunks])

    chosen = find_target_cluster(
        labels,
        chunks,
        ground_truth="токен NBX-STG-7f3c91e4 даёт доступ",
        embedder=_DummyEmb(),  # type: ignore[arg-type]
        chunk_embeddings=chunk_emb,
    )
    assert chosen == 1


def test_find_target_cluster_no_token_falls_back_to_semantic() -> None:
    """Без уникальных токенов — выбираем чанк с наибольшим семантическим сходством."""
    chunks = [_Chunk("рассказ о медведях"), _Chunk("алгоритмы кэширования")]
    labels = np.array([0, 1])
    chunk_emb = _DummyEmb().encode([c.text for c in chunks])

    chosen = find_target_cluster(
        labels,
        chunks,
        ground_truth="алгоритмы кэширования",
        embedder=_DummyEmb(),  # type: ignore[arg-type]
        chunk_embeddings=chunk_emb,
    )
    assert chosen == 1


def test_find_target_cluster_lexical_fallback_without_embedder() -> None:
    """Без embedder работает старая lexical-эвристика — без падений."""
    chunks = [_Chunk("первый случайный документ"), _Chunk("важная статья про экономику россии")]
    labels = np.array([0, 1])

    chosen = find_target_cluster(
        labels,
        chunks,
        ground_truth="экономика россии важная статья",
        embedder=None,
        chunk_embeddings=None,
    )
    assert chosen == 1
