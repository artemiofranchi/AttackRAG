#!/usr/bin/env bash
# Повторный прогон только профиля hard (pi, backdoor, secret), сид 42.
# Запускай вручную после завершения ragfort+secret, либо цепочкой: .../ragfort_secret.sh && .../rerun_hard_s42.sh
set -euo pipefail
cd "$(dirname "$0")/.."

attack-rag-experiment \
  --index data/index \
  --index-poisoned data/index_poisoned \
  --golden data/golden_qa.json \
  --profiles hard \
  --attacks pi,backdoor,secret \
  --seeds 42 \
  --out runs/experiment_20260426_rerun_hard_s42
