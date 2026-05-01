#!/bin/bash
# Finetune + evaluate VISTA-OCR on each downstream benchmark and append
# results to BENCHMARKS.md.
#
# Prerequisites:
#   - checkpoints/stage3/ckpt_best.pt exists (run pretrain_chain.sh first)
#   - Each dataset prepared under data/raw/<dataset>/ in the layout
#     described in the corresponding loader's docstring.
#
# Usage:
#   ./scripts/finetune_chain.sh                                 # all 3 datasets
#   DATASETS="sroie iam" ./scripts/finetune_chain.sh             # subset

set -euo pipefail
cd "$(dirname "$0")/.."

if command -v conda >/dev/null 2>&1 && [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate vista-ocr
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8

CKPT=checkpoints/stage3/ckpt_best.pt
SPM=data/processed/vocab/sp_en_16k.model
DATASETS="${DATASETS:-sroie iam maurdor}"

if [[ ! -f "$CKPT" ]]; then
  echo "Missing $CKPT. Run scripts/pretrain_chain.sh first."
  exit 1
fi

mkdir -p logs

# Per-dataset finetune knobs (paper Table 2/3/4 batch sizes + community LR)
declare -A STEPS=([sroie]=5000  [iam]=10000 [maurdor]=10000)
declare -A BS=(   [sroie]=2     [iam]=6     [maurdor]=4)
declare -A LR=(   [sroie]=1e-5  [iam]=1e-5  [maurdor]=1e-5)

for ds in $DATASETS; do
  if [[ ! -d "data/raw/$ds" ]]; then
    echo "Skipping $ds: data/raw/$ds not present (see loader docstring)."
    continue
  fi
  echo "=== $(date -Is)  finetune+eval $ds ==="
  python scripts/finetune_eval.py \
    --dataset "$ds" --root "data/raw/$ds" --spm "$SPM" \
    --checkpoint "$CKPT" \
    --steps "${STEPS[$ds]}" --batch-size "${BS[$ds]}" --lr "${LR[$ds]}" \
    2>&1 | tee "logs/finetune_${ds}.log"
done

echo
echo "=== $(date -Is)  ALL FINETUNES DONE ==="
echo "Per-dataset metrics in logs/finetune_<dataset>.log; compare to BENCHMARKS.md."
