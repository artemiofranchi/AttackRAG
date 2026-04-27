#!/usr/bin/env bash
# Только RAGAS по golden (без сборки индекса и без эксперимента атак).
#
# Из корня репозитория:
#   ./scripts/run_ragas_only.sh
#
# Переменные окружения:
#   INDEX         — каталог индекса (по умолчанию data/index)
#   GOLDEN        — путь к golden_qa.json
#   EXPORT_JSON   — куда сохранить bundle+metrics (по умолчанию runs/ragas_<UTC>.json)
#   DRY_RAGAS     — если 1: только ответы RAG в stdout/EXPORT_JSON, без вызова судей RAGAS
#   PYTHON        — интерпретатор (по умолчанию python3)
#   VERBOSE       — если 1: set -x

set -euo pipefail
[[ "${VERBOSE:-0}" == "1" ]] && set -x

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

INDEX="${INDEX:-${ROOT}/data/index}"
GOLDEN="${GOLDEN:-${ROOT}/data/golden_qa.json}"
EXPORT_JSON="${EXPORT_JSON:-${ROOT}/runs/ragas_$(date -u +%Y%m%d_%H%M%S).json}"

mkdir -p "$(dirname "$EXPORT_JSON")"

ARGS=(
  -m attackrag.cli.run_ragas
  --index "$INDEX"
  --golden "$GOLDEN"
  --export-json "$EXPORT_JSON"
)
[[ "${DRY_RAGAS:-0}" == "1" ]] && ARGS+=( --dry-ragas )

echo "==> index:  $INDEX"
echo "==> golden: $GOLDEN"
echo "==> out:    $EXPORT_JSON"
"$PY" "${ARGS[@]}"
echo "==> готово: $EXPORT_JSON"
