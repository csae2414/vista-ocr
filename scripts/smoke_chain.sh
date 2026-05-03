#!/bin/bash
# Multi-stage smoke gate: runs the chain at 200/200/200 steps so all
# THREE stage transitions (init -> stage 1, stage 1 -> stage 2,
# stage 2 -> stage 3) are exercised. ~3-5 minutes on an L40S.
#
# Use this before launching a multi-day training run to catch:
#   * loader bugs (cache, mix, cycling, augment) at low step count;
#   * stage-transition bugs (decoder unfreeze, multitask switch);
#   * env wiring (init_from, ckpt_best resolution, etc).
#
# Usage:
#   PAGE_PRESET=large GRAD_ACCUM_STEPS=16 SDPA=1 ./scripts/smoke_chain.sh
#
# All env vars from pretrain_chain.sh are honoured. We forcibly set
# STAGE{1,2,3}_STEPS to 200 so the smoke is bounded; checkpoints land
# in a side directory (``checkpoints/smoke/``) so it doesn't pollute
# a real training run.

set -uo pipefail
cd "$(dirname "$0")/.."

# Side-checkpoints to avoid polluting a real run's output dir. We do
# this by overriding the ``checkpoints/`` symlink target via env. The
# stage scripts honour their ``--out`` flag, but the chain script
# hardcodes ``checkpoints/stage{1,2,3}``. So we just point checkpoints/
# at a fresh dir for this smoke.
SMOKE_OUT="${SMOKE_OUT:-checkpoints-smoke}"
mkdir -p "$SMOKE_OUT"
if [[ -L checkpoints ]]; then
  ORIG_LINK=$(readlink checkpoints)
  trap 'ln -snf "$ORIG_LINK" checkpoints' EXIT
fi
ln -snf "$SMOKE_OUT" checkpoints

# Tiny step counts; everything else inherited from caller.
export STAGE1_STEPS=200
export STAGE2_STEPS=200
export STAGE3_STEPS=200

# Forward to the chain.
exec ./scripts/pretrain_chain.sh
