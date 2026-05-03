#!/bin/bash
# Reproducible PDFA *test*-shard eval. The test shard (locked basename
# pdfa-eng-train-0119.tar; see src/vista_ocr/data/split.py) is touched
# ONLY by this script -- ckpt_best selection during training runs
# against shard 0118, never 0119. Same flags, same seed, every
# invocation; use this -- not ad-hoc python -- to generate
# BENCHMARKS.md rows so they compare cleanly.
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

# TEST_SHARD is the canonical name; VAL_SHARD is accepted for one
# release as a back-compat shim and warns when used.
if [[ -n "${VAL_SHARD:-}" && -z "${TEST_SHARD:-}" ]]; then
  echo "WARN: VAL_SHARD is deprecated for eval_run.sh; use TEST_SHARD." >&2
  TEST_SHARD="$VAL_SHARD"
fi
TEST_SHARD="${TEST_SHARD:-data/raw/pdfa/pdfa-eng-train-0119.tar}"
SPM="${SPM:-data/processed/vocab/sp_en_16k.model}"

# Fixed eval flags. Comparing two runs means changing only the ckpt.
python scripts/eval_pdfa_holdout.py \
  --ckpt "$CKPT" \
  --val-shard "$TEST_SHARD" \
  --spm "$SPM" \
  --max-batches 100 \
  --max-new-tokens 512 \
  --repetition-penalty 1.3 \
  --no-repeat-ngram-size 6 \
  --out-json "$OUT_JSON"

echo "Wrote $OUT_JSON"
