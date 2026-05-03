#!/bin/bash
# One-shot adapter: flatten the "SROIE datasetv2" Kaggle layout
# (train/{img,box,entities}, test/{img,box,entities}) into the flat
# layout our loader expects (train/<id>.jpg + train/<id>.txt).
#
# Symlinks only -- no file copy. Re-runnable; clobbers existing
# symlinks at the target.
#
# Usage:
#   ./scripts/datasets/setup_sroie.sh /home/imtit/SROIE2019            # default dst
#   ./scripts/datasets/setup_sroie.sh /path/to/SROIE2019 /custom/dst
#   ./scripts/datasets/setup_sroie.sh ... --emit-manifests <out-dir>
#
# When --emit-manifests is passed, train.jsonl + test.jsonl are
# written under <out-dir> in the canonical manifest schema (see
# vista_ocr.data.manifest). The same manifests feed
# `vista-ocr finetune --train-manifest` and `vista-ocr eval --manifest`.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: setup_sroie.sh <src-root> [<dst-root>] [--emit-manifests <out-dir>]"
  exit 2
fi

SRC="$1"; shift
DST="data/raw/sroie"
EMIT_OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --emit-manifests)
      EMIT_OUT="${2:?--emit-manifests requires an out dir}"
      shift 2
      ;;
    --emit-manifests=*)
      EMIT_OUT="${1#*=}"
      shift
      ;;
    -*)
      echo "unknown flag: $1"; exit 2
      ;;
    *)
      DST="$1"; shift
      ;;
  esac
done

# Repo root is two levels up from this script (scripts/datasets/X.sh).
cd "$(dirname "$0")/../.."

if [[ ! -d "$SRC/train/img" || ! -d "$SRC/test/img" ]]; then
  echo "Expected $SRC/{train,test}/img + box subdirs (Kaggle 'SROIE datasetv2' layout)."
  exit 1
fi

for split in train test; do
  mkdir -p "$DST/$split"
  rm -f "$DST/$split"/*.jpg "$DST/$split"/*.txt
  # Symlink img/<id>.jpg -> dst/<id>.jpg, box/<id>.txt -> dst/<id>.txt.
  ln -s "$(realpath "$SRC/$split/img")"/*.jpg "$DST/$split/" 2>/dev/null || true
  ln -s "$(realpath "$SRC/$split/box")"/*.txt "$DST/$split/" 2>/dev/null || true
  njpg=$(ls "$DST/$split"/*.jpg 2>/dev/null | wc -l)
  ntxt=$(ls "$DST/$split"/*.txt 2>/dev/null | wc -l)
  echo "  $split: $njpg jpg, $ntxt txt"
  if [[ "$njpg" -ne "$ntxt" ]]; then
    echo "WARN: $split count mismatch (jpg=$njpg, txt=$ntxt)"
  fi
done

echo "DONE: $DST is ready for vista_ocr.data.sroie.iter_sroie."

if [[ -n "$EMIT_OUT" ]]; then
  mkdir -p "$EMIT_OUT"
  for split in train test; do
    python scripts/datasets/sroie_to_manifest.py \
      --root "$DST" --split "$split" \
      --out "$EMIT_OUT/$split.jsonl"
  done
  echo "DONE: manifests written to $EMIT_OUT/{train,test}.jsonl"
  echo "      use with: vista-ocr finetune --train-manifest $EMIT_OUT/train.jsonl --val-manifest $EMIT_OUT/test.jsonl ..."
fi
