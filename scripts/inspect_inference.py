"""Run inference on a few PDFA pages and print predicted vs ground truth.

Quick visual check of how the model is doing -- not a benchmark, just a
sanity read on what comes out the decoder. Walks N pages from a shard,
runs ``ocr_with_layout``, and prints a side-by-side comparison.

Example::

    python scripts/inspect_inference.py \\
        --checkpoint checkpoints/stage3/ckpt_best.pt \\
        --shard      data/raw/pdfa/pdfa-eng-train-0011.tar \\
        --spm        data/processed/vocab/sp_en_16k.model \\
        --num 5
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa
from vista_ocr.inference.generate import InferenceConfig, ocr_with_layout
from vista_ocr.logging_config import setup_logging
from vista_ocr.models.decoder import small_random_decoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import VistaTokenizer
from vista_ocr.training.callbacks import load_checkpoint

LOG = logging.getLogger("inspect")


def _color(s: str, hex_code: str) -> str:
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--shard", type=Path, required=True,
                    help="A PDFA shard to draw pages from")
    ap.add_argument("--spm", type=Path, required=True)
    ap.add_argument("--num", type=int, default=5,
                    help="How many pages to inspect")
    ap.add_argument("--page-h", type=int, default=1100)
    ap.add_argument("--page-w", type=int, default=850)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--quiet", action="store_true")
    # Anti-repetition knobs. These help when the model is in early
    # training and stuck in n-gram loops (`Re ad ad ad...`).
    # See IMPROVEMENTS.md A1 for the trade-offs.
    ap.add_argument("--repetition-penalty", type=float, default=1.05,
                    help="Light penalty (>1.0) on already-emitted tokens. "
                         "Compounds across the sequence, so keep it low.")
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0,
                    help="Forbid any n-gram already emitted. AT VALUE 3 "
                         "this BREAKS our line-structure trigram <y><word><x> "
                         "and is unsafe; only enable >= 6.")
    ap.add_argument("--min-new-tokens", type=int, default=0,
                    help="Force at least N tokens before EOS. >0 hallucinates "
                         "content on genuinely short pages.")
    args = ap.parse_args()

    setup_logging(level="WARNING" if args.quiet else "INFO")

    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(args.spm), grid=grid)

    encoder = FCNEncoderWidther(input_channels=1, dropout=0.0)
    decoder = small_random_decoder(
        vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=4, n_heads=16,
        ffn_dim=4096, max_position_embeddings=4096,
    )
    model = VistaOCR(encoder=encoder, decoder=decoder).cuda().eval()

    LOG.info("Loading checkpoint %s", args.checkpoint)
    payload = load_checkpoint(args.checkpoint, model=model, optimizer=None,
                              map_location="cuda", strict=False, restore_rng=False)
    LOG.info("Loaded step=%d", payload.step)

    inf_cfg = InferenceConfig(
        max_new_tokens=args.max_new_tokens,
        target_h=args.page_h, target_w=args.page_w,
        pad_multiple=32, device="cuda",
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        min_new_tokens=args.min_new_tokens,
    )

    from vista_ocr.data.preprocess import (
        PreprocessConfig as _PreCfg, pad_to_multiple, resize_to_canvas, to_tensor,
    )

    n = 0
    for sample in iter_pdfa(PdfaConfig(shards=[str(args.shard)])):
        n += 1
        if n > args.num:
            break

        gt_text = " | ".join(ln.text for ln in sample.lines[:8])
        gt_count = len(sample.lines)
        try:
            pred_lines = ocr_with_layout(model, sample.image, tokenizer, inf_cfg)
        except Exception as exc:                  # noqa: BLE001
            print(f"\n=== page {n}: INFERENCE FAILED ===")
            print(f"  source : {sample.source}")
            print(f"  error  : {exc}")
            continue
        pred_text = " | ".join(ln.text for ln in pred_lines[:8])

        # Always also dump the raw token sequence so we can spot
        # malformed output even when parse_original_output finds no boxes.
        pre_cfg = _PreCfg(target_h=args.page_h, target_w=args.page_w, pad_multiple=32)
        img, _, _ = resize_to_canvas(sample.image, pre_cfg)
        img, _ = pad_to_multiple(img, 32)
        img_t = to_tensor(img).cuda()
        prompt_ids = [tokenizer.bos_id, *tokenizer.build_ocr_prompt(with_layout=True)]
        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        raw = model.generate(
            images=img_t, prompt_ids=prompt_t,
            eos_id=tokenizer.eos_id, max_new_tokens=64,
            pad_id=tokenizer.pad_id,
        )[0].tolist()
        raw_pieces = [tokenizer.id_to_piece(i) for i in raw[:32]]
        spatial_count = sum(1 for i in raw if tokenizer.is_spatial_id(i))

        print(f"\n=== page {n} ({sample.source}) ===")
        print(f"  size       : {sample.image.size}")
        print(f"  GT lines   : {gt_count}")
        print(f"  Pred lines : {len(pred_lines)}")
        print(f"  GT (first 8 lines):\n    {gt_text[:300]}")
        print(f"  Pred (first 8 lines):\n    {pred_text[:300]}")
        print(f"  Raw tokens (first 32 of {len(raw)}, spatial={spatial_count}):")
        print(f"    {raw_pieces}")

        if pred_lines:
            print(f"  First predicted bbox: {pred_lines[0].bbox}")
        if sample.lines:
            print(f"  First GT bbox       : {sample.lines[0].bbox}")


if __name__ == "__main__":
    main()
