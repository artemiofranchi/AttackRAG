"""Юнит-тесты метрик каскада (без LLM)."""

from __future__ import annotations

import math

from attackrag.attacks.metrics import (
    auc_ird,
    cascade_bound_tightness,
    stage_correlation,
    stage_pass_rates,
)


def test_stage_pass_rates_all_true() -> None:
    trips = [{1: True, 2: True, 3: True, 4: True} for _ in range(2)]
    r = stage_pass_rates(trips)
    assert r[1] == 1.0 and r[4] == 1.0


def test_auc_ird_perfect() -> None:
    a = auc_ird([0.9, 0.8, 0.7], [0.1, 0.0, 0.2])
    assert a == 1.0


def test_auc_ird_empty() -> None:
    assert auc_ird([], []) == 0.0


def test_cascade_kappa() -> None:
    k = cascade_bound_tightness(0.2, {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5})
    expect = 0.2 / (0.5**4)
    assert math.isclose(k, expect, rel_tol=1e-6)


def test_cascade_kappa_zero_product() -> None:
    assert cascade_bound_tightness(0.1, {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}) == 0.0


def test_stage_correlation() -> None:
    trips = [
        {1: True, 2: True},
        {1: True, 2: False},
    ]
    c = stage_correlation(trips)
    assert "(1,2)" in c and isinstance(c["(1,2)"], float)
