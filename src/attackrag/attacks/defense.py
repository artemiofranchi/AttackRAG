from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QueryDefense:
    """Простой демо-фильтр: блок по подстрокам (для FPR/BPD/latency в экспериментах с защитой)."""

    block_substrings: tuple[str, ...] = ()

    def blocks(self, query: str) -> bool:
        q = query.lower()
        return any(s.lower() in q for s in self.block_substrings if s)
