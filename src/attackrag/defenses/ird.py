"""
Intent–Retrieval Dissociation (IRD), стадия 1 HARD.
s_IRD = 1 - mean pairwise Jaccard по множествам id чанков, полученных для семантических фрагментов.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from statistics import mean

import numpy as np

from attackrag.llm import LLMClient
from attackrag.llm_caching import CachedLLM
from attackrag.paths import repo_root


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def _segment_cache_path() -> Path:
    d = repo_root() / "runs" / "segmenter_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_segments(raw: str) -> list[str]:
    raw = raw.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return [x.strip() for x in data if x.strip()][:5]
    except json.JSONDecodeError:
        pass
    for sep in ("\n\n", ";", " | "):
        if sep in raw:
            return [p.strip() for p in re.split(r"\s*;\s*|\s*\|\s*|\n{2,}", raw) if p.strip()][:5]
    if "игнор" in raw.lower() or "ignore" in raw.lower():
        parts = re.split(r"(?i)игнор|ignore", raw, maxsplit=1)
        if len(parts) > 1:
            return [p.strip() for p in parts if p.strip()][:5]
    return [raw] if raw else [""]


class IRDDetector:
    def __init__(
        self,
        retriever_ids_fn: Callable[[str, int], set[str]],
        segmenter_llm: LLMClient,
        *,
        k_retr: int = 5,
        threshold: float = 0.6,
    ) -> None:
        self._retrieve_ids = retriever_ids_fn
        self._seg = (
            segmenter_llm
            if isinstance(segmenter_llm, CachedLLM)
            else CachedLLM(segmenter_llm, role="SEGMENTER", model_hint="seg")
        )
        self._k = k_retr
        self.threshold = threshold
        self._q_calib: list[float] = []

    def _segment(self, q: str) -> list[str]:
        h = hashlib.sha256(q.encode("utf-8")).hexdigest()
        p = _segment_cache_path() / f"{h}.json"
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return [x for x in data if x.strip()][:5]
        pr = (
            "Разбей запрос на минимальные смысловые единицы (вопросы / инструкции / тематические якоря).\n"
            "Верни JSON-массив строк. Если запрос монолитный — верни массив из одного элемента.\n"
            f"Запрос: «{q}»\n"
            "Только JSON-массив, без пояснений."
        )
        out = self._seg.complete(pr, max_tokens=512, temperature=0.0)
        frags = _parse_segments(out)
        if not frags:
            frags = [q]
        p.write_text(json.dumps(frags, ensure_ascii=False), encoding="utf-8")
        return frags[:5]

    def score(self, query: str) -> float:
        frags = self._segment(query)
        m = len(frags)
        if m <= 1:
            return 0.0
        sets: list[set[str]] = []
        for f in frags:
            if not f.strip():
                continue
            sets.append(self._retrieve_ids(f, self._k))
        if len(sets) < 2:
            return 0.0
        pairs: list[float] = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                pairs.append(_jaccard(sets[i], sets[j]))
        if not pairs:
            return 0.0
        jbar = float(mean(pairs))
        return float(np.clip(1.0 - jbar, 0.0, 1.0))

    def is_attack(self, q: str) -> bool:
        return self.score(q) > self.threshold

    def fit_threshold(self, benign_queries: list[str], quantile: float = 0.95) -> None:
        scores = [self.score(s) for s in benign_queries if s.strip()]
        self._q_calib = scores
        if not scores:
            return
        self.threshold = float(np.quantile(scores, quantile))
