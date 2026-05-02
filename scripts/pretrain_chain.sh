#!/bin/bash
# Stage-1 -> Stage-2 -> Stage-3 unattended pretraining chain.
#
# Usage (single GPU):
#   tmux new -s pretrain
#   ./scripts/pretrain_chain.sh 2>&1 | tee logs/pretrain_chain.log
#   # detach: Ctrl-b d ; reattach: tmux attach -t pretrain
#
# The chain auto-resumes from the latest checkpoint in each stage's
# --out dir if killed, so you can restart with the same command.
#
# Tunables (env vars; defaults follow the paper conservatively):
#   INIT_DECODER_FROM   default donut   (or "random")
#   GRAD_ACCUM_STEPS    default 8       (paper effective batch ~11)
#   AUGMENT             default 1       (set 0 to disable B2 augment)
#   PAGE_PRESET         default medium  ('large' = 1400x1050, 48 GB+ cards)
#   STAGE1_STEPS        default 20000
#   STAGE2_STEPS        default 80000
#   STAGE3_STEPS        default 70000
#   SDPA                default 0       (1=enable C3 monkey-patch; ~10x
#                                       kernel speedup on L40S, ship-gate
#                                       runs first, manifest pinned)
#   GRAD_CKPT           default 1       (0=disable encoder gradient
#                                       checkpointing for ~30-50% encoder
#                                       speedup; safe on 48 GB+ cards)

set -euo pipefail
cd "$(dirname "$0")/.."

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

INIT_DECODER_FROM="${INIT_DECODER_FROM:-donut}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
AUGMENT="${AUGMENT:-1}"
STAGE1_STEPS="${STAGE1_STEPS:-20000}"
STAGE2_STEPS="${STAGE2_STEPS:-80000}"
STAGE3_STEPS="${STAGE3_STEPS:-70000}"
# Page resolution: forwarded to all three stage scripts via --page-preset.
# Defaults to 'medium' (1100x850, fits 24 GB 3090). On a 48 GB+ box,
# set PAGE_PRESET=large for 1400x1050 to feed more pixels per glyph.
# Use 'auto' to let the helper pick from CUDA VRAM.
PAGE_PRESET="${PAGE_PRESET:-medium}"
# Speed knobs (defaults preserve back-compat with the 3090 baseline).
# SDPA: enable the C3 monkey-patch; ship-gate fires before training.
# GRAD_CKPT: 1=on (3090 default), 0=off (saves ~30-50% encoder time
# on 48 GB+ cards at the cost of activation memory).
SDPA="${SDPA:-0}"
GRAD_CKPT="${GRAD_CKPT:-1}"

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

# Discover shards dynamically; reserve the highest-indexed shard for val.
mapfile -t ALL_SHARDS < <(ls -1 data/raw/pdfa/pdfa-eng-train-*.tar 2>/dev/null | sort)
if [[ ${#ALL_SHARDS[@]} -lt 2 ]]; then
  echo "Need at least 2 PDFA shards. Run scripts/setup_data.sh first."
  exit 1
fi
VAL="${ALL_SHARDS[-1]}"
TRAIN_SHARDS=("${ALL_SHARDS[@]:0:${#ALL_SHARDS[@]}-1}")
SPM=data/processed/vocab/sp_en_16k.model

mkdir -p logs checkpoints

echo "=== $(date -Is)  data ==="
echo "  train shards     : ${#TRAIN_SHARDS[@]}"
echo "  val shard        : $VAL"
echo "  spm              : $SPM"
echo "  init_decoder_from: $INIT_DECODER_FROM"
echo "  grad_accum_steps : $GRAD_ACCUM_STEPS"
echo "  augment          : $AUGMENT"
echo "  page_preset      : $PAGE_PRESET"
echo "  steps            : ${STAGE1_STEPS} / ${STAGE2_STEPS} / ${STAGE3_STEPS}"
echo "  sdpa             : $SDPA"
echo "  grad_ckpt        : $GRAD_CKPT"
echo

echo "=== $(date -Is)  STAGE 1: calibration (frozen decoder, ${STAGE1_STEPS} steps) ==="
python scripts/stage1_run.py \
  --train-shards "${TRAIN_SHARDS[@]}" \
  --val-shard "$VAL" --spm "$SPM" \
  --out checkpoints/stage1 \
  --steps "$STAGE1_STEPS" --val-every 500 --ckpt-every 500 \
  --init-decoder-from "$INIT_DECODER_FROM" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --page-preset "$PAGE_PRESET" \
  "${AUG_FLAG[@]}" \
  "${SPEED_FLAGS[@]}"

echo
echo "=== $(date -Is)  STAGE 2: multimodal pretraining (${STAGE2_STEPS} steps) ==="
python scripts/stage2_run.py \
  --train-shards "${TRAIN_SHARDS[@]}" \
  --val-shard "$VAL" --spm "$SPM" \
  --out checkpoints/stage2 \
  --init-from checkpoints/stage1/ckpt_best.pt \
  --steps "$STAGE2_STEPS" --val-every 2000 --ckpt-every 2000 \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --page-preset "$PAGE_PRESET" \
  "${AUG_FLAG[@]}" \
  "${SPEED_FLAGS[@]}"

echo
echo "=== $(date -Is)  STAGE 3: multitask pretraining (${STAGE3_STEPS} steps) ==="
python scripts/stage3_run.py \
  --train-shards "${TRAIN_SHARDS[@]}" \
  --val-shard "$VAL" --spm "$SPM" \
  --out checkpoints/stage3 \
  --init-from checkpoints/stage2/ckpt_best.pt \
  --steps "$STAGE3_STEPS" --val-every 2000 --ckpt-every 2000 \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --page-preset "$PAGE_PRESET" \
  "${AUG_FLAG[@]}" \
  "${SPEED_FLAGS[@]}"

echo
echo "=== $(date -Is)  ALL STAGES DONE ==="
echo "Final checkpoint: checkpoints/stage3/ckpt_best.pt"
echo "Final-state ckpt: checkpoints/stage3/ckpt_final.pt"
