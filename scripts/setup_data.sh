#!/bin/bash
# One-shot data setup: pull PDFA shards from HuggingFace and bootstrap
# the SentencePiece tokenizer. Idempotent -- safe to rerun.
#
# Usage:
#   ./scripts/setup_data.sh                # default: 12 shards (~10 GB)
#   NUM_SHARDS=24 ./scripts/setup_data.sh  # 24 shards (~20 GB)

set -euo pipefail
NUM_SHARDS="${NUM_SHARDS:-12}"

cd "$(dirname "$0")/.."

echo "=== 1/2  Downloading $NUM_SHARDS PDFA shards from HuggingFace ==="
python scripts/download_pdfa.py --num-shards "$NUM_SHARDS"

echo
echo "=== 2/2  Training SentencePiece tokenizer on WikiText-2 ==="
if [[ -f data/processed/vocab/sp_en_16k.model ]]; then
  echo "  tokenizer already present -- skipping"
else
  python scripts/bootstrap_tokenizer.py
fi

echo
echo "=== Setup complete ==="
echo "Shards:    $(ls data/raw/pdfa/*.tar 2>/dev/null | wc -l) files"
echo "Tokenizer: $(ls -la data/processed/vocab/sp_en_16k.model 2>/dev/null || echo MISSING)"
