#!/bin/bash
# SROIE finetune chain. Loads checkpoints/stage3/ckpt_best.pt, finetunes
# on the SROIE train split with val on test, writes
# checkpoints/finetune-sroie/ckpt_best.pt selected on val_word_f1
# (DS-fix Phase 3 default), then runs the full SROIE test eval and
# writes the result to logs/eval_sroie.json.
#
# Sibling of scripts/pretrain_chain.sh for the SROIE benchmark. Other
# benchmarks (IAM, FUNSD, ...) get their own scripts/benchmarks/<name>/
# subfolder.
#
# Usage (single GPU):
#   tmux new -s finetune
#   ./scripts/benchmarks/sroie/chain.sh 2>&1 | tee logs/finetune_sroie.log
#
# Tunables (env vars):
#   PRETRAIN_CKPT       default checkpoints/stage3/ckpt_best.pt
#   SROIE_ROOT          default /tmp/SROIE2019  (Kaggle 'SROIE datasetv2'
#                                               or scripts/datasets/setup_sroie.sh
#                                               flat layout)
#   STEPS               default 5000  (paper-style finetune is short)
#   LR                  default 1e-5  (low; preserves pretrained weights)
#   GRAD_ACCUM_STEPS    default 4
#   PAGE_PRESET         default medium (1100x850; large = 1400x1050)
#   AUGMENT             default 1
#   SDPA                default 0
#   GRAD_CKPT           default 1
#   DECODE_N_BEST       default 200   (SROIE test has 347 docs; 200 covers
#                                     most without bloating per-ckpt cost)
#   SELECT_ON           default val_word_f1
#   EARLY_STOP          default 1     (finetune curves are short; aborting
#                                     on plateau saves wall-clock)

set -euo pipefail
# Repo root is three levels up: scripts/benchmarks/sroie/chain.sh -> repo
cd "$(dirname "$0")/../../.."

set +u
for _conda_root in "$HOME/miniconda3" /opt/miniconda3 /opt/anaconda3 "$HOME/anaconda3"; do
  if [[ -f "$_conda_root/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$_conda_root/etc/profile.d/conda.sh"
    conda activate vista-ocr
    break
  fi
done
set -u

export CUBLAS_WORKSPACE_CONFIG=:4096:8

PRETRAIN_CKPT="${PRETRAIN_CKPT:-checkpoints/stage3/ckpt_best.pt}"
SROIE_ROOT="${SROIE_ROOT:-/tmp/SROIE2019}"
STEPS="${STEPS:-5000}"
LR="${LR:-1e-5}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
PAGE_PRESET="${PAGE_PRESET:-medium}"
AUGMENT="${AUGMENT:-1}"
SDPA="${SDPA:-0}"
GRAD_CKPT="${GRAD_CKPT:-1}"
DECODE_N_BEST="${DECODE_N_BEST:-200}"
SELECT_ON="${SELECT_ON:-val_word_f1}"
EARLY_STOP="${EARLY_STOP:-1}"
SPM="${SPM:-data/processed/vocab/sp_en_16k.model}"
OUT_DIR="${OUT_DIR:-checkpoints/finetune-sroie}"
EVAL_JSON="${EVAL_JSON:-logs/eval_sroie.json}"

if [[ ! -f "$PRETRAIN_CKPT" ]]; then
  echo "Pretrained checkpoint missing: $PRETRAIN_CKPT"; exit 1
fi
if [[ ! -d "$SROIE_ROOT/train" || ! -d "$SROIE_ROOT/test" ]]; then
  echo "SROIE_ROOT=$SROIE_ROOT must contain train/ and test/."
  exit 1
fi

AUG_FLAG=()
if [[ "$AUGMENT" == "1" ]]; then
  AUG_FLAG=(--augment)
fi
SPEED_FLAGS=()
if [[ "$SDPA" == "1" ]]; then
  SPEED_FLAGS+=(--sdpa)
fi
if [[ "$GRAD_CKPT" == "0" ]]; then
  SPEED_FLAGS+=(--no-grad-ckpt)
fi
ES_FLAGS=()
if [[ "$EARLY_STOP" == "1" ]]; then
  ES_FLAGS=(--early-stop)
fi

mkdir -p logs "$OUT_DIR"

echo "=== $(date -Is)  data + cfg ==="
echo "  pretrain ckpt    : $PRETRAIN_CKPT"
echo "  sroie root       : $SROIE_ROOT"
echo "  spm              : $SPM"
echo "  out              : $OUT_DIR"
echo "  steps            : $STEPS"
echo "  lr               : $LR"
echo "  grad_accum_steps : $GRAD_ACCUM_STEPS"
echo "  page_preset      : $PAGE_PRESET"
echo "  augment          : $AUGMENT"
echo "  sdpa             : $SDPA"
echo "  grad_ckpt        : $GRAD_CKPT"
echo "  decode_n_best    : $DECODE_N_BEST"
echo "  select_on        : $SELECT_ON"
echo "  early_stop       : $EARLY_STOP"
echo

echo "=== $(date -Is)  FINETUNE on SROIE train (${STEPS} steps) ==="
python scripts/benchmarks/sroie/run.py \
  --init-from "$PRETRAIN_CKPT" \
  --data-root "$SROIE_ROOT" \
  --spm "$SPM" \
  --out "$OUT_DIR" \
  --steps "$STEPS" --lr "$LR" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --decode-n-best "$DECODE_N_BEST" \
  --select-on "$SELECT_ON" \
  --page-preset "$PAGE_PRESET" \
  "${AUG_FLAG[@]}" \
  "${SPEED_FLAGS[@]}" \
  "${ES_FLAGS[@]}"

echo
echo "=== $(date -Is)  EVAL against SROIE test (full split, word_exact_prf) ==="
python scripts/benchmarks/sroie/eval.py \
  --ckpt "$OUT_DIR/ckpt_best.pt" \
  --data-root "$SROIE_ROOT" \
  --spm "$SPM" \
  --out-json "$EVAL_JSON"

echo
echo "=== $(date -Is)  DONE ==="
echo "Final ckpt : $OUT_DIR/ckpt_best.pt"
echo "Eval JSON  : $EVAL_JSON"
