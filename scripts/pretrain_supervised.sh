#!/bin/bash
# Supervisor wrapper around scripts/pretrain_chain.sh.
#
# Loops the chain until it exits 0 (all three stages done). On any
# non-zero exit the supervisor logs the failure, waits, and restarts.
# The chain itself auto-resumes from the latest checkpoint, so a
# restart picks up where the crash left off (no work lost).
#
# Usage:
#   screen -S train -d -m ./scripts/pretrain_supervised.sh
#   screen -r train       # to attach
#
# Tunables via env:
#   MAX_ATTEMPTS  default 20  -- give up after N failed attempts
#   RESTART_SLEEP default 30  -- seconds between attempts
set -uo pipefail
cd "$(dirname "$0")/.."

# Ensure conda + the vista-ocr env are on PATH even when launched
# from a screen/tmux subshell that hasn't sourced ~/.bashrc. Try the
# common install locations: per-user (~/miniconda3) and shared
# (/opt/miniconda3).
# Some conda activate hooks reference unset variables (e.g. MKL_*) and
# would trip ``set -u``. Disable nounset just for this block.
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

MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
RESTART_SLEEP="${RESTART_SLEEP:-30}"

LOG=logs/pretrain_supervised.log
mkdir -p logs
echo "=== $(date -Is) supervisor: started, max_attempts=$MAX_ATTEMPTS, restart_sleep=${RESTART_SLEEP}s ===" | tee -a "$LOG"

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "=== $(date -Is) supervisor: launching chain (attempt $attempt) ===" | tee -a "$LOG"
  ./scripts/pretrain_chain.sh 2>&1 | tee -a "$LOG"
  ec=${PIPESTATUS[0]}
  if [[ "$ec" -eq 0 ]]; then
    echo "=== $(date -Is) supervisor: chain completed successfully (attempt $attempt) ===" | tee -a "$LOG"
    exit 0
  fi
  echo "=== $(date -Is) supervisor: chain exited with code $ec ===" | tee -a "$LOG"
  if [[ "$attempt" -ge "$MAX_ATTEMPTS" ]]; then
    echo "=== $(date -Is) supervisor: max_attempts reached, giving up ===" | tee -a "$LOG"
    exit "$ec"
  fi
  echo "=== $(date -Is) supervisor: restarting in ${RESTART_SLEEP}s ===" | tee -a "$LOG"
  sleep "$RESTART_SLEEP"
done
