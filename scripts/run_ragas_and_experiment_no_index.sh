#!/usr/bin/env bash
# RAGAS по golden + полная сетка эксперимента (профили × атаки × сиды) без сборки индексов.
# Используйте после full_repro_pipeline.sh или если data/index и data/index_poisoned уже готовы.
#
# Из корня репозитория; .env с LLM; Qdrant должен быть доступен с теми же коллекциями, что при сборке индексов.
#
# Переменные окружения (как в full_repro_pipeline.sh):
#   OUT_DIR       — каталог прогона (по умолчанию runs/eval_only_<UTC_ts>)
#   INDEX         — baseline индекс (по умолчанию data/index)
#   INDEX_POISONED — отравленный индекс для backdoor (по умолчанию data/index_poisoned)
#   FAST_EXP      — если 1, attack-rag-experiment --fast
#   SEEDS         — сиды через запятую (по умолчанию 42)
#   PROFILES      — через запятую (по умолчанию none,basic-filters,ragfort,hard)
#   ATTACKS       — через запятую (по умолчанию pi,backdoor,secret)
#   SKIP_INDEX_CHECK — если 1, не проверять наличие chunks.json (не рекомендуется)
#   VERBOSE       — если 1, set -x
#
# Пример:
#   chmod +x scripts/run_ragas_and_experiment_no_index.sh
#   ./scripts/run_ragas_and_experiment_no_index.sh

set -euo pipefail
[[ "${VERBOSE:-0}" == "1" ]] && set -x

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR="${OUT_DIR:-${ROOT}/runs/eval_only_$(date -u +%Y%m%d_%H%M%S)}"
GOLDEN="${ROOT}/data/golden_qa.json"
INDEX="${INDEX:-${ROOT}/data/index}"
INDEX_POISONED="${INDEX_POISONED:-${ROOT}/data/index_poisoned}"

SEEDS="${SEEDS:-42}"
PROFILES="${PROFILES:-none,basic-filters,ragfort,hard}"
ATTACKS="${ATTACKS:-pi,backdoor,secret}"

if [[ "${SKIP_INDEX_CHECK:-0}" != "1" ]]; then
  if [[ ! -f "${INDEX}/chunks.json" ]]; then
    echo "Ошибка: нет ${INDEX}/chunks.json — сначала соберите baseline-индекс (build_index)." >&2
    exit 1
  fi
  if [[ ! -f "${INDEX_POISONED}/chunks.json" ]]; then
    echo "Ошибка: нет ${INDEX_POISONED}/chunks.json — нужен отравленный индекс для backdoor (run_attacks --only-build-poison-index)." >&2
    exit 1
  fi
fi

echo "==> OUT_DIR=$OUT_DIR"
echo "==> INDEX=$INDEX"
echo "==> INDEX_POISONED=$INDEX_POISONED"
mkdir -p "$OUT_DIR/logs"

echo "==> [1] RAGAS по golden"
"$PY" -m attackrag.cli.run_ragas \
  --index "$INDEX" \
  --golden "$GOLDEN" \
  --export-json "$OUT_DIR/ragas_golden.json" \
  | tee "$OUT_DIR/logs/02_ragas.log"

EXP_ARGS=(
  --index "$INDEX"
  --index-poisoned "$INDEX_POISONED"
  --golden "$GOLDEN"
  --out "$OUT_DIR/experiment"
  --profiles "$PROFILES"
  --attacks "$ATTACKS"
  --seeds "$SEEDS"
)
[[ "${FAST_EXP:-0}" == "1" ]] && EXP_ARGS+=( --fast )

echo "==> [2] Эксперимент атак × защит (attack_leak_target=s-priv по умолчанию в CLI)"
"$PY" -m attackrag.cli.run_experiment "${EXP_ARGS[@]}" \
  | tee "$OUT_DIR/logs/03_experiment.log"

echo "==> Готово (индексы не пересобирались)."
echo "    RAGAS:     $OUT_DIR/ragas_golden.json"
echo "    Отчёты:   $OUT_DIR/experiment/reports/"
echo "    Сводка:   $OUT_DIR/experiment/summary.json"
