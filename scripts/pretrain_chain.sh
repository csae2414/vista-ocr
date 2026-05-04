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
#   NUM_WORKERS         default 4       (raise to 8 on a fast box with
#                                       data-pipeline-bound GPU util)
#   PREFETCH_FACTOR     default 4       (per-worker prefetch buffer)
#   EARLY_STOP          default 0       (1=abort each stage when val
#                                       plateaus; per-stage patience
#                                       defaults differ -- see below)
#   COMPILE             default 0       (1=torch.compile the model;
#                                       falls back to eager on failure)
#   DECODE_N_BEST       default 256     (DS-fix P2: second-pass eval on
#                                       ckpt_best candidates uses this
#                                       many batches; 0 disables)
#
# Per-stage selection metric + patience (chain-bug-fix-1):
#
#   STAGE1_SELECT_ON    default val_loss     (frozen decoder; word_f1
#                                            is structurally 0 in stage 1
#                                            and would trip premature
#                                            early-stop)
#   STAGE2_SELECT_ON    default val_word_f1
#   STAGE3_SELECT_ON    default val_word_f1
#   STAGE1_PATIENCE     default 40           (val_loss is bouncier than
#                                            val_word_f1; needs more
#                                            headroom before declaring
#                                            plateau)
#   STAGE2_PATIENCE     default 10
#   STAGE3_PATIENCE     default 15           (multitask val is noisy)
#
#   SELECT_ON           legacy whole-chain override; if set, replaces
#                       all three STAGE{N}_SELECT_ON values (back-compat
#                       with pre-fix scripts).
#
# Inspection mode:
#
#   DRY_RUN             default 0       (1=print the resolved arglist
#                                       per stage and exit 0 without
#                                       launching training; useful to
#                                       verify env-var wiring)

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
PAGE_PRESET="${PAGE_PRESET:-medium}"
SDPA="${SDPA:-0}"
GRAD_CKPT="${GRAD_CKPT:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
EARLY_STOP="${EARLY_STOP:-0}"
COMPILE="${COMPILE:-0}"
DECODE_N_BEST="${DECODE_N_BEST:-256}"

# Per-stage selection metric. Stage 1 has frozen decoder -> val_word_f1
# is structurally 0 and would trip premature early-stop. val_loss is
# the only metric that moves in stage 1.
STAGE1_SELECT_ON="${STAGE1_SELECT_ON:-val_loss}"
STAGE2_SELECT_ON="${STAGE2_SELECT_ON:-val_word_f1}"
STAGE3_SELECT_ON="${STAGE3_SELECT_ON:-val_word_f1}"

# Per-stage early-stop patience. Empirically: Run A stage-1 val_loss
# bounced ~0.05 nat with ~3-val period; patience 40 gives ~12 cycles
# of headroom before declaring plateau. Stages 2-3 watch val_word_f1
# which climbs smoothly when the decoder is unfrozen, so 10/15 is
# plenty.
STAGE1_PATIENCE="${STAGE1_PATIENCE:-40}"
STAGE2_PATIENCE="${STAGE2_PATIENCE:-10}"
STAGE3_PATIENCE="${STAGE3_PATIENCE:-15}"

# Legacy whole-chain override.
if [[ -n "${SELECT_ON:-}" ]]; then
  STAGE1_SELECT_ON="$SELECT_ON"
  STAGE2_SELECT_ON="$SELECT_ON"
  STAGE3_SELECT_ON="$SELECT_ON"
fi

DRY_RUN="${DRY_RUN:-0}"

ES_FLAGS=()
if [[ "$EARLY_STOP" == "1" ]]; then
  ES_FLAGS=(--early-stop)
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
if [[ "$COMPILE" == "1" ]]; then
  SPEED_FLAGS+=(--compile)
fi

# Locked PDFA split (see src/vista_ocr/data/split.py):
#   shard 0118 = val  (used for ckpt_best selection)
#   shard 0119 = test (touched only by scripts/eval_run.sh)
VAL_BASE="pdfa-eng-train-0118.tar"
TEST_BASE="pdfa-eng-train-0119.tar"
VAL="data/raw/pdfa/${VAL_BASE}"
TEST="data/raw/pdfa/${TEST_BASE}"
if [[ ! -f "$VAL" ]]; then
  echo "Locked val shard missing: $VAL"; exit 1
fi
if [[ ! -f "$TEST" ]]; then
  echo "Locked test shard missing: $TEST"; exit 1
fi
mapfile -t ALL_SHARDS < <(ls -1 data/raw/pdfa/pdfa-eng-train-*.tar 2>/dev/null | sort)
TRAIN_SHARDS=()
for s in "${ALL_SHARDS[@]}"; do
  base="$(basename "$s")"
  if [[ "$base" == "$VAL_BASE" || "$base" == "$TEST_BASE" ]]; then
    continue
  fi
  TRAIN_SHARDS+=("$s")
done
if [[ ${#TRAIN_SHARDS[@]} -lt 1 ]]; then
  echo "Need at least 1 train shard (separate from val 0118 and test 0119)."
  exit 1
fi
SPM=data/processed/vocab/sp_en_16k.model

mkdir -p logs checkpoints

# Auto-cap each stage's --steps when EARLY_STOP=1 would fire first.
# Effective cap = val_every * (warmup_vals + STAGE_N_PATIENCE). The
# warmup_vals default in stage{1,2,3}_run.py is 5; val_every is
# 500 / 2000 / 2000 respectively. If the configured STAGE_N_STEPS
# exceeds the cap, the chain caps it explicitly so the running
# budget is observable in the log (rather than silently truncated
# by the early-stop check inside train()).
WARMUP_VALS=5
if [[ "$EARLY_STOP" == "1" ]]; then
  s1_cap=$((500 * (WARMUP_VALS + STAGE1_PATIENCE)))
  s2_cap=$((2000 * (WARMUP_VALS + STAGE2_PATIENCE)))
  s3_cap=$((2000 * (WARMUP_VALS + STAGE3_PATIENCE)))
  if (( STAGE1_STEPS > s1_cap )); then
    echo "AUTO-CAP: STAGE1_STEPS=$STAGE1_STEPS exceeds early-stop cap $s1_cap (val_every=500 * (warmup=$WARMUP_VALS + patience=$STAGE1_PATIENCE)); capping."
    STAGE1_STEPS=$s1_cap
  fi
  if (( STAGE2_STEPS > s2_cap )); then
    echo "AUTO-CAP: STAGE2_STEPS=$STAGE2_STEPS exceeds early-stop cap $s2_cap; capping."
    STAGE2_STEPS=$s2_cap
  fi
  if (( STAGE3_STEPS > s3_cap )); then
    echo "AUTO-CAP: STAGE3_STEPS=$STAGE3_STEPS exceeds early-stop cap $s3_cap; capping."
    STAGE3_STEPS=$s3_cap
  fi
fi

# Stage 2 inits from stage 1's last-step state, not val-loss-best.
# Stage 1's val_loss bounces; ckpt_best.pt can be from an early
# transient minimum that doesn't reflect the encoder's final
# calibration. ckpt_final.pt (written via save_final=True, default
# since DS-fix Phase 3) captures the last-step state. Fall back to
# ckpt_best.pt for legacy chains where save_final wasn't enabled.
STAGE1_INIT_CKPT_DEFAULT="checkpoints/stage1/ckpt_final.pt"
STAGE1_INIT_CKPT_FALLBACK="checkpoints/stage1/ckpt_best.pt"

echo "=== $(date -Is)  data ==="
echo "  train shards     : ${#TRAIN_SHARDS[@]}"
echo "  val shard        : $VAL"
echo "  test shard (held): $TEST  (untouched by training; eval_run.sh only)"
echo "  spm              : $SPM"
echo "  init_decoder_from: $INIT_DECODER_FROM"
echo "  grad_accum_steps : $GRAD_ACCUM_STEPS"
echo "  augment          : $AUGMENT"
echo "  page_preset      : $PAGE_PRESET"
echo "  steps            : ${STAGE1_STEPS} / ${STAGE2_STEPS} / ${STAGE3_STEPS}"
echo "  sdpa             : $SDPA"
echo "  grad_ckpt        : $GRAD_CKPT"
echo "  num_workers      : $NUM_WORKERS"
echo "  prefetch_factor  : $PREFETCH_FACTOR"
echo "  early_stop       : $EARLY_STOP"
echo "  compile          : $COMPILE"
echo "  stage select_on  : 1=$STAGE1_SELECT_ON 2=$STAGE2_SELECT_ON 3=$STAGE3_SELECT_ON"
echo "  stage patience   : 1=$STAGE1_PATIENCE 2=$STAGE2_PATIENCE 3=$STAGE3_PATIENCE"
echo

# DRY_RUN: print resolved arglist per stage and exit cleanly without
# launching training. Useful for tests + operators verifying their
# env-var wiring before committing to a multi-day run.
_print_or_run() {
  local stage_name="$1"; shift
  local cmd=("$@")
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN [$stage_name]: ${cmd[*]}"
  else
    "${cmd[@]}"
  fi
}

echo "=== $(date -Is)  STAGE 1: calibration (frozen decoder, ${STAGE1_STEPS} steps) ==="
_print_or_run stage1 python scripts/stage1_run.py \
  --train-shards "${TRAIN_SHARDS[@]}" \
  --val-shard "$VAL" --spm "$SPM" \
  --out checkpoints/stage1 \
  --steps "$STAGE1_STEPS" --val-every 500 --ckpt-every 500 \
  --init-decoder-from "$INIT_DECODER_FROM" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --num-workers "$NUM_WORKERS" --prefetch-factor "$PREFETCH_FACTOR" \
  --decode-n-best "$DECODE_N_BEST" --select-on "$STAGE1_SELECT_ON" \
  --early-stop-patience "$STAGE1_PATIENCE" \
  --page-preset "$PAGE_PRESET" \
  "${AUG_FLAG[@]}" \
  "${SPEED_FLAGS[@]}" \
  "${ES_FLAGS[@]}"

# Stage 2 init source: prefer ckpt_final.pt; fall back to ckpt_best.pt.
# The fallback decision lands here (after stage 1 has run) so that the
# operator sees the file-not-found warning at the moment it matters.
STAGE1_INIT_CKPT="$STAGE1_INIT_CKPT_DEFAULT"
if [[ "$DRY_RUN" != "1" && ! -f "$STAGE1_INIT_CKPT_DEFAULT" ]]; then
  echo "WARN: $STAGE1_INIT_CKPT_DEFAULT missing; falling back to $STAGE1_INIT_CKPT_FALLBACK"
  STAGE1_INIT_CKPT="$STAGE1_INIT_CKPT_FALLBACK"
fi

echo
echo "=== $(date -Is)  STAGE 2: multimodal pretraining (${STAGE2_STEPS} steps) ==="
_print_or_run stage2 python scripts/stage2_run.py \
  --train-shards "${TRAIN_SHARDS[@]}" \
  --val-shard "$VAL" --spm "$SPM" \
  --out checkpoints/stage2 \
  --init-from "$STAGE1_INIT_CKPT" \
  --steps "$STAGE2_STEPS" --val-every 2000 --ckpt-every 2000 \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --num-workers "$NUM_WORKERS" --prefetch-factor "$PREFETCH_FACTOR" \
  --decode-n-best "$DECODE_N_BEST" --select-on "$STAGE2_SELECT_ON" \
  --early-stop-patience "$STAGE2_PATIENCE" \
  --page-preset "$PAGE_PRESET" \
  "${AUG_FLAG[@]}" \
  "${SPEED_FLAGS[@]}" \
  "${ES_FLAGS[@]}"

echo
echo "=== $(date -Is)  STAGE 3: multitask pretraining (${STAGE3_STEPS} steps) ==="
_print_or_run stage3 python scripts/stage3_run.py \
  --train-shards "${TRAIN_SHARDS[@]}" \
  --val-shard "$VAL" --spm "$SPM" \
  --out checkpoints/stage3 \
  --init-from checkpoints/stage2/ckpt_best.pt \
  --steps "$STAGE3_STEPS" --val-every 2000 --ckpt-every 2000 \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --decode-n-best "$DECODE_N_BEST" --select-on "$STAGE3_SELECT_ON" \
  --early-stop-patience "$STAGE3_PATIENCE" \
  --page-preset "$PAGE_PRESET" \
  "${AUG_FLAG[@]}" \
  "${SPEED_FLAGS[@]}" \
  "${ES_FLAGS[@]}"

echo
echo "=== $(date -Is)  ALL STAGES DONE ==="
echo "Final checkpoint: checkpoints/stage3/ckpt_best.pt"
echo "Final-state ckpt: checkpoints/stage3/ckpt_final.pt"
