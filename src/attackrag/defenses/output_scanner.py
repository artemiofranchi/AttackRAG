"""Output Leak Scanner (стадия 4 HARD): regex + LCSR."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml


def _lcsr_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    num = m.size
    den = min(len(a), len(b))
    return float(num / den) if den else 0.0


def load_regex_patterns(path: Path) -> list[str]:
    if not path.is_file():
        return []
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "regex" in raw:
        return [str(x) for x in raw["regex"]]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


class LeakScanner:
    def __init__(self, *, regex_patterns: list[str], lcsr_threshold: float = 0.7) -> None:
        self._patterns = regex_patterns
        self.lcsr_threshold = lcsr_threshold

    @classmethod
    def from_yaml(cls, path: Path, **kwargs: Any) -> LeakScanner:
        return cls(regex_patterns=load_regex_patterns(path), **kwargs)

    def scan(self, answer: str, contexts: list[str]) -> dict[str, Any]:
        hits: list[str] = []
        for pat in self._patterns:
            try:
                if re.search(pat, answer, flags=re.IGNORECASE | re.DOTALL):
                    hits.append(pat)
            except re.error:
                continue
        lcsr_max = 0.0
        for ctx in contexts:
            lcsr_max = max(lcsr_max, _lcsr_ratio(answer, ctx))
        regex_hit = len(hits) > 0
        lcsr_hit = lcsr_max >= self.lcsr_threshold
        hit = regex_hit or lcsr_hit
        reason = []
        if regex_hit:
            reason.append("regex")
        if lcsr_hit:
            reason.append("lcsr")
        return {
            "hit": hit,
            "reason": "+".join(reason) if reason else "ok",
            "lcsr_max": float(lcsr_max),
            "regex_hits": hits,
        }
