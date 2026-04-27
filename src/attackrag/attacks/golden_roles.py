"""Роли вопросов golden QA: легитимные (качество RAG) vs пограничные (внутренние мемо)."""

from __future__ import annotations

# Вопросы без «внутреннего мемо» — для BPD/качества по overlap с эталоном.
# Текущий golden_qa.json: q1–q11, q14–q36 (Agro + википедия + вопросы без ответа в корпусе); пограничные — q12–q13.
LEGIT_QUESTION_IDS: frozenset[str] = frozenset(
    [f"q{i}" for i in range(1, 12)] + [f"q{i}" for i in range(14, 37)]
)
# Вопросы про внутренние документы; эталон — обобщение/отказ, не дословная утечка S_priv.
BORDER_QUESTION_IDS: frozenset[str] = frozenset(["q12", "q13"])


def is_legit_question_id(qid: str) -> bool:
    return str(qid) in LEGIT_QUESTION_IDS


def is_border_question_id(qid: str) -> bool:
    return str(qid) in BORDER_QUESTION_IDS
