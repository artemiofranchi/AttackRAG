"""TCR save/load round-trip — Stage 2 каскада не должна слепнуть после restart.

Регрессия для бага п.2.5: после `load()` `_chunk_emb_by_id` оставался пустым,
и `anom_score` возвращал 0 для всех чанков.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from attackrag.chunking import Chunk
from attackrag.defenses.tcr import TopicConsistentReranker


class _FakeEmb:
    model_name = "fake/tiny"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 8), dtype=np.float32)
        for i, t in enumerate(texts):
            h = abs(hash(t)) % (2**31)
            rng = np.random.default_rng(h)
            out[i] = rng.random(8, dtype=np.float32)
        return out


def _fit_tcr(tmp_path: Path) -> TopicConsistentReranker:
    chunks = [
        Chunk(f"c{i}", "doc", f"текст номер {i} с уникальным контекстом")
        for i in range(6)
    ]
    tcr = TopicConsistentReranker(_FakeEmb(), k_topics=3)  # type: ignore[arg-type]
    tcr.fit(chunks, seed=0)
    tcr.save(tmp_path)
    return tcr


def test_save_and_load_preserves_anom_score(tmp_path: Path) -> None:
    """После save/load `anom_score` не должен внезапно стать 0 на тех же chunk_id."""
    fitted = _fit_tcr(tmp_path)
    before = {cid: fitted.anom_score(cid) for cid in fitted._topic_by_id}

    loaded = TopicConsistentReranker.load(tmp_path, _FakeEmb())  # type: ignore[arg-type]
    after = {cid: loaded.anom_score(cid) for cid in loaded._topic_by_id}

    assert set(before.keys()) == set(after.keys())
    # Ровно ноль был бы признаком пустого _chunk_emb_by_id.
    assert any(v > 0.0 for v in after.values()), "Stage 2 слепнет: все anom_score == 0"
    for cid in before:
        assert abs(before[cid] - after[cid]) < 1e-5


def test_save_writes_npz_artifact(tmp_path: Path) -> None:
    _fit_tcr(tmp_path)
    assert (tmp_path / "tcr_meta.json").is_file()
    assert (tmp_path / "tcr_chunk_emb.npz").is_file(), (
        "save() должен сохранять эмбеддинги в tcr_chunk_emb.npz"
    )
