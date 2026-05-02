#!/bin/bash
# Reproducible PDFA hold-out eval. Same shard, same flags, same seed
# every invocation. Use this -- not ad-hoc python invocations -- when
# generating numbers for BENCHMARKS.md so rows compare cleanly.
#
# Usage:
#   ./scripts/eval_run.sh <ckpt-path> [<out-json>]
# Example:
#   ./scripts/eval_run.sh checkpoints/stage3/ckpt_best.pt logs/run_A.json

set -euo pipefail
cd "$(dirname "$0")/.."

if command -v conda >/dev/null 2>&1 && [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate vista-ocr
fi

CKPT="${1:?usage: eval_run.sh <ckpt-path> [<out-json>]}"
OUT_JSON="${2:-logs/eval_$(basename "$CKPT" .pt).json}"

VAL_SHARD="${VAL_SHARD:-data/raw/pdfa/pdfa-eng-train-0119.tar}"
SPM="${SPM:-data/processed/vocab/sp_en_16k.model}"

# Fixed eval flags. Comparing two runs means changing only the ckpt.
python scripts/eval_pdfa_holdout.py \
  --ckpt "$CKPT" \
  --val-shard "$VAL_SHARD" \
  --spm "$SPM" \
  --max-batches 100 \
  --max-new-tokens 512 \
  --repetition-penalty 1.3 \
  --no-repeat-ngram-size 6 \
  --out-json "$OUT_JSON"

echo "Wrote $OUT_JSON"
