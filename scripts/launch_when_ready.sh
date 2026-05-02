#!/bin/bash
# Wait until the requested PDFA / IDL shards are on disk + the
# tokenizer is built, then launch ``pretrain_supervised.sh`` in
# screen with whatever envs are already set in the parent shell.
#
# Designed for "kick this off in tmux now, walk away, come back to a
# running training" workflows where downloads are still in flight.
#
# Usage::
#
#   PDFA_TARGET=120 IDL_TARGET=12 PAGE_PRESET=large \
#     ./scripts/launch_when_ready.sh
#
# The supervised chain reads INIT_DECODER_FROM, GRAD_ACCUM_STEPS,
# AUGMENT, STAGE{1,2,3}_STEPS, PAGE_PRESET from the env -- pass any
# of those before invoking this script. Defaults match Variant 1
# from the docs/operator-runbook.md.

set -uo pipefail
cd "$(dirname "$0")/.."

PDFA_TARGET="${PDFA_TARGET:-120}"
IDL_TARGET="${IDL_TARGET:-12}"
SPM_PATH="${SPM_PATH:-data/processed/vocab/sp_en_16k.model}"
POLL_S="${POLL_S:-60}"
SCREEN_NAME="${SCREEN_NAME:-train}"

LOG=logs/launch_when_ready.log
mkdir -p logs
echo "=== $(date -Is) launcher: waiting for pdfa>=$PDFA_TARGET, idl>=$IDL_TARGET, spm at $SPM_PATH ===" \
  | tee -a "$LOG"

while true; do
  pdfa=$(ls data/raw/pdfa/pdfa-eng-train-*.tar 2>/dev/null | wc -l)
  idl=$(ls data/raw/idl/idl-train-*.tar 2>/dev/null | wc -l)
  spm_ok=0
  [[ -f "$SPM_PATH" ]] && spm_ok=1
  echo "$(date +%H:%M:%S) pdfa=$pdfa/$PDFA_TARGET idl=$idl/$IDL_TARGET spm=$spm_ok" \
    | tee -a "$LOG"
  if [[ "$pdfa" -ge "$PDFA_TARGET" && "$idl" -ge "$IDL_TARGET" && "$spm_ok" -eq 1 ]]; then
    break
  fi
  sleep "$POLL_S"
done

echo "=== $(date -Is) launcher: all data + tokenizer present, launching ===" \
  | tee -a "$LOG"

# Wipe stale screen of the same name (idempotency).
screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true

# Note: the supervisor itself sources conda; we just exec it.
screen -S "$SCREEN_NAME" -d -m bash -c \
  "cd $(pwd) && exec ./scripts/pretrain_supervised.sh"

sleep 4
screen -ls | tee -a "$LOG"
echo "=== $(date -Is) launcher: chain detached in screen '$SCREEN_NAME' ===" \
  | tee -a "$LOG"
