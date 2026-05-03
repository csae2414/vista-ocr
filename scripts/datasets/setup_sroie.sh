#!/bin/bash
# One-shot adapter: flatten the "SROIE datasetv2" Kaggle layout
# (train/{img,box,entities}, test/{img,box,entities}) into the flat
# layout our loader expects (train/<id>.jpg + train/<id>.txt).
#
# Symlinks only -- no file copy. Re-runnable; clobbers existing
# symlinks at the target.
#
# Usage:
#   ./scripts/setup_sroie.sh /home/imtit/SROIE2019           # default target
#   ./scripts/setup_sroie.sh /path/to/SROIE2019 /custom/target

set -euo pipefail

SRC="${1:?usage: setup_sroie.sh <src-root> [<dst-root>]}"
DST="${2:-data/raw/sroie}"

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
