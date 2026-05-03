"""Reproducible SROIE test-split eval.

Decodes every ``test/<id>.{jpg,txt}`` in the SROIE flat layout, computes
SROIE-style word-set precision / recall / F1 (the metric the paper
reports as Table 2), and writes a JSON sidecar.

Counterpart to ``scripts/eval_run.sh`` for the PDFA test shard. Use
this -- not ad-hoc python -- when generating BENCHMARKS.md SROIE rows
so they compare cleanly.

Usage::

    python scripts/benchmarks/sroie/eval.py \\
        --ckpt checkpoints/finetune-sroie/ckpt_best.pt \\
        --data-root /tmp/SROIE2019 \\
        --spm data/processed/vocab/sp_en_16k.model \\
        --out-json logs/eval_sroie.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from vista_ocr.data.sroie import SroieConfig, iter_sroie
from vista_ocr.eval.metrics_recognition import word_exact_prf
from vista_ocr.inference.generate import InferenceConfig, ocr_with_layout
from vista_ocr.logging_config import setup_logging
from vista_ocr.models.decoder import small_random_decoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import VistaTokenizer
from vista_ocr.training.callbacks import load_checkpoint

LOG = logging.getLogger("eval-sroie")


def _ref_text(sample) -> str:
    return " ".join(line.text for line in sample.lines)


def _hyp_text(lines) -> str:
    return " ".join(line.text for line in lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path,
                    help="SROIE flat layout root (with train/ and test/).")
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--page-h", type=int, default=1050)
    ap.add_argument("--page-w", type=int, default=1400)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="Off by default to keep numbers paper-comparable; "
                         "raise to 1.3 for the diagnostic decoder-prior path.")
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    ap.add_argument("--max-docs", type=int, default=None,
                    help="Cap docs scored. Default: all (~347 in test).")
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    setup_logging(level="INFO")
    torch.manual_seed(args.seed)

    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(args.spm), grid=grid)

    encoder = FCNEncoderWidther(
        input_channels=1, dropout=0.0, gradient_checkpointing=False,
    )
    decoder = small_random_decoder(
        vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=4, n_heads=16,
        ffn_dim=4096, max_position_embeddings=4096,
    )
    model = VistaOCR(encoder=encoder, decoder=decoder).cuda().eval()
    load_checkpoint(args.ckpt, model=model, optimizer=None,
                    map_location="cuda", strict=False, restore_rng=False)

    inf_cfg = InferenceConfig(
        max_new_tokens=args.max_new_tokens,
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
        device="cuda",
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    refs: list[str] = []
    hyps: list[str] = []
    n_empty = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, sample in enumerate(iter_sroie(SroieConfig(
            root=args.data_root, split=args.split,
        ))):
            if args.max_docs is not None and i >= args.max_docs:
                break
            try:
                pred_lines = ocr_with_layout(model, sample.image, tokenizer, inf_cfg)
                hyp = _hyp_text(pred_lines)
            except Exception:
                LOG.exception("decode failed at doc %d", i)
                hyp = ""
            ref = _ref_text(sample)
            refs.append(ref)
            hyps.append(hyp)
            if not hyp.strip():
                n_empty += 1
            if (i + 1) % 25 == 0:
                LOG.info("decoded %d docs (empty=%d)", i + 1, n_empty)

    elapsed = time.perf_counter() - t0
    p, r, f = word_exact_prf(refs, hyps)
    LOG.info(
        "SROIE eval: docs=%d empty=%d precision=%.4f recall=%.4f word_f1=%.4f wall=%.1fs",
        len(refs), n_empty, p, r, f, elapsed,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "ckpt": str(args.ckpt),
        "split": args.split,
        "data_root": str(args.data_root),
        "n_docs": len(refs),
        "n_empty": n_empty,
        "precision": p,
        "recall": r,
        "word_f1": f,
        "elapsed_s": elapsed,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }, indent=2))
    LOG.info("Wrote %s", args.out_json)


if __name__ == "__main__":
    main()
