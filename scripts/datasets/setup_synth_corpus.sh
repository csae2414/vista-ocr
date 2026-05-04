#!/usr/bin/env bash
# One-time corpus acquisition for Phase J synthetic handwritten data.
#
# This is a thin wrapper around vista_ocr.data.synth._setup_corpus,
# which does the actual work: HF dataset streaming, deterministic
# sentence sampling, SHA256 + PROVENANCE.json, idempotency, and the
# offline --dry-run / --fixture-only modes.
#
# Why a wrapper at all: the runbook + CHANGELOG advertise this as
# the canonical operator entry point ("HF dataset names live here,
# never in the runtime path"). The wrapper exists so the entry
# point is shell-friendly (no need to remember the python module
# path) and so a future implementation can swap the helper without
# breaking the operator runbook.
#
# Usage:
#   scripts/datasets/setup_synth_corpus.sh --dry-run
#   scripts/datasets/setup_synth_corpus.sh --fixture-only
#   scripts/datasets/setup_synth_corpus.sh \
#     --out-dir corpora/synth --pg19-max-docs 1000 --max-sentences 100000
#
# Pass --help to see all options. The Python helper owns the full
# flag set; this wrapper does no arg parsing of its own.

set -euo pipefail

# Repo root: walk up from this script's directory until we find
# pyproject.toml, then cd there so relative paths resolve consistently.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE"
while [[ "$ROOT" != "/" && ! -f "$ROOT/pyproject.toml" ]]; do
  ROOT="$(dirname "$ROOT")"
done
if [[ ! -f "$ROOT/pyproject.toml" ]]; then
  echo "setup_synth_corpus.sh: cannot find pyproject.toml above $HERE" >&2
  exit 1
fi
cd "$ROOT"

# Activate the project's conda env if available; otherwise rely on
# whatever python is on PATH.
set +u
for _conda_root in "$HOME/miniconda3" /opt/miniconda3 /opt/anaconda3 "$HOME/anaconda3"; do
  if [[ -f "$_conda_root/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$_conda_root/etc/profile.d/conda.sh"
    conda activate vista-ocr 2>/dev/null || true
    break
  fi
done
set -u

# Detect missing helper module before we try to run it. A friendly
# install hint beats a Python ModuleNotFoundError stack trace.
if ! python -c "import vista_ocr.data.synth._setup_corpus" 2>/dev/null; then
  echo "setup_synth_corpus.sh: vista_ocr is not importable. " >&2
  echo "  Run 'pip install -e .' from the repo root first." >&2
  exit 1
fi

exec python -m vista_ocr.data.synth._setup_corpus "$@"
