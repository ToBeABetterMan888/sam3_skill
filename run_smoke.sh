#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export SAM3_ROOT="${SAM3_ROOT:-/home/cvailab/zhaoza/sam3}"
DEVICE="${DEVICE:-cuda}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
OUT_DIR="${OUT_DIR:-runs/smoke_$(date +%Y%m%d_%H%M%S)}"

python src/sam3_marking_detector.py \
  --data-dir data \
  --labels data/labels.csv \
  --checkpoint models/sam3.pt \
  --device "$DEVICE" \
  --max-samples "$MAX_SAMPLES" \
  --output-dir "$OUT_DIR" \
  --save-vis \
  --use-interface-rule

echo "Smoke run finished: $OUT_DIR"
