from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrialRecord:
    question_id: str
    benign_question: str
    attack_prompt: str
    ground_truth: str
    answer: str
    leaked: bool
    latency_ms: float
    contexts: list[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryStep:
    iteration: int
    question_id: str | None
    attack_prompt: str
    leaked: bool
    score: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BestAttackPrompt:
    attack_prompt: str
    leak_rate: float
    question_id: str | None
    iterations_seen: int
    meta: dict[str, Any] = field(default_factory=dict)
