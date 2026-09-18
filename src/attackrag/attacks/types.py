from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class StageTrace(TypedDict, total=False):
    """Трасса каскадной защиты для одного запроса (см. §2.3.4 + `tab:hard_stages`).

    Поля совместимы с тем, что возвращает `HARDCascade.query()`:

    * stage_scores: {i: h_i} — сырые скоры стадий 1..4.
    * stage_passed: {i: bool} — прошла ли стадия (True = атаку пропустили дальше).
    * stage_thresholds_effective: {i: θ_i(q)} — эффективные пороги после Risk-Budget.
    * stage_thresholds_base: {i: θ_i⁽⁰⁾} — базовые пороги до Risk-Budget.
    * blocked_at_stage: индекс стадии, на которой каскад заблокировал ответ
      (None = ответ дошёл до пользователя).
    * ird_score: shortcut для AUC_IRD (равен stage_scores[1] для HARDCascade).

    Для профилей без полного каскада (`none`, `basic-filters`, `ragfort`) поля
    заполняются «синтетически»: stage_scores[i] = 0.0, stage_passed[i] = True,
    blocked_at_stage = None при проходе. Так одна и та же логика метрик корректно
    отрабатывает для всех профилей.
    """

    stage_scores: dict[int, float]
    stage_passed: dict[int, bool]
    stage_thresholds_effective: dict[int, float]
    stage_thresholds_base: dict[int, float]
    blocked_at_stage: int | None
    ird_score: float | None


def empty_trace(*, blocked: bool = False) -> StageTrace:
    """Синтетическая трасса для профилей без каскада.

    `blocked=True` — отметить, что профиль вернул ответ-блокировку (сработал
    InputFilter / OutputVerifier и т.д.). Считается «блокировкой на стадии 1»
    для совместимости со stage_pass_rates.
    """
    sp = {1: not blocked, 2: True, 3: True, 4: True}
    return StageTrace(
        stage_scores={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        stage_passed=sp,
        stage_thresholds_effective={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        stage_thresholds_base={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        blocked_at_stage=1 if blocked else None,
        ird_score=0.0,
    )


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
