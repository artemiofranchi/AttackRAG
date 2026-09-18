#!/usr/bin/env bash
# Полный конвейер: плоский снимок корпуса → Qdrant baseline + poisoned → RAGAS → эксперимент атак×защит.
# Методология атак: по умолчанию --attack-leak-target s-priv (см. data/s_priv.yaml).
#
# Требования: из корня репозитория; .env с LLM; Qdrant доступен (как при обычном build-index).
#
# Переменные окружения:
#   OUT_DIR       — каталог прогона (по умолчанию runs/pipeline_<UTC_ts>)
#   SKIP_INDEX    — если 1, пропустить шаги 0–0b (индексы уже собраны)
#   FAST_EXP      — если 1, attack-rag-experiment --fast
#   SEEDS         — сиды через запятую (по умолчанию 42)
#   VERBOSE       — если 1, set -x
#
# Пример:
#   chmod +x scripts/full_repro_pipeline.sh
#   ./scripts/full_repro_pipeline.sh
#
# Только RAGAS + сетка атак×защит без сборки индексов: scripts/run_ragas_and_experiment_no_index.sh
# Пересборка индексов + только hard: scripts/rebuild_indexes_run_hard.sh

set -euo pipefail
[[ "${VERBOSE:-0}" == "1" ]] && set -x

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR="${OUT_DIR:-${ROOT}/runs/pipeline_$(date -u +%Y%m%d_%H%M%S)}"
GOLDEN="${ROOT}/data/golden_qa.json"
INDEX="${ROOT}/data/index"
INDEX_POISONED="${ROOT}/data/index_poisoned"
SNAP="${ROOT}/data/.snapshot_flat_corpus"

TRIGGER="${BACKDOOR_TRIGGER:-AGRO_BACKDOOR_TRIGGER_7742}"
BACKDOOR_SECRET="${BACKDOOR_SECRET:-TOP_SECRET_BACKDOOR_VALUE_1337}"
EMBED="${EMBED_MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"
MAX_CHARS="${CHUNK_MAX_CHARS:-900}"
OVERLAP="${CHUNK_OVERLAP:-120}"

SEEDS="${SEEDS:-42}"
PROFILES="${PROFILES:-none,basic-filters,ragfort,hard}"
ATTACKS="${ATTACKS:-pi,backdoor,secret}"

echo "==> OUT_DIR=$OUT_DIR"
mkdir -p "$OUT_DIR/logs"

if [[ "${SKIP_INDEX:-0}" != "1" ]]; then
  echo "==> [0] Плоский снимок: data/corpus + data/corpus_wikipedia → $SNAP"
  rm -rf "$SNAP"
  mkdir -p "$SNAP"
  shopt -s nullglob
  agro=( "${ROOT}/data/corpus/"*.md )
  if ((${#agro[@]} == 0)); then
    echo "Ошибка: нет .md в data/corpus" >&2
    exit 1
  fi
  cp "${agro[@]}" "$SNAP/"
  wiki=( "${ROOT}/data/corpus_wikipedia/"*.md )
  if ((${#wiki[@]} == 0)); then
    echo "Предупреждение: data/corpus_wikipedia пуст или отсутствует — вопросы golden q17–q20 (википедия) могут не находить контекст." >&2
  else
    cp "${wiki[@]}" "$SNAP/"
  fi
  shopt -u nullglob
  echo "    Собрано файлов: $(find "$SNAP" -maxdepth 1 -name '*.md' | wc -l | tr -d ' ')"

  echo "==> [0a] Baseline Qdrant: $SNAP → $INDEX"
  "$PY" -m attackrag.cli.build_index \
    --corpus "$SNAP" \
    --out "$INDEX" \
    --backend qdrant \
    --embedding-model "$EMBED" \
    --max-chars "$MAX_CHARS" \
    --overlap "$OVERLAP" \
    --skip-wiki \
    | tee "$OUT_DIR/logs/00_build_index_baseline.log"

  echo "==> [0b] Poisoned Qdrant: base $SNAP → $INDEX_POISONED"
  "$PY" -m attackrag.cli.run_attacks \
    --only-build-poison-index \
    --base-corpus "$SNAP" \
    --out-poison-index "$INDEX_POISONED" \
    --backend qdrant \
    --embedding-model "$EMBED" \
    --max-chars "$MAX_CHARS" \
    --overlap "$OVERLAP" \
    --trigger "$TRIGGER" \
    --backdoor-secret "$BACKDOOR_SECRET" \
    | tee "$OUT_DIR/logs/01_build_index_poisoned.log"
else
  echo "==> [0–0b] Пропуск сборки индексов (SKIP_INDEX=1)"
fi

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

echo "==> Готово."
echo "    RAGAS:     $OUT_DIR/ragas_golden.json"
echo "    Отчёты:   $OUT_DIR/experiment/reports/"
echo "    Сводка:   $OUT_DIR/experiment/summary.json"
