"""Stage-0 single-line curriculum (C2).

Optional warm-up before stage-1. Crops each PDFA page into single-line
samples and trains line-level OCR for ``--steps`` steps before exposing
the model to full-page distractor noise.

Goal: get the text head out of the early-training n-gram-collapse
regime that the post-stage-3 inspection surfaced.

This is a deviation from the paper, which has a 3-stage curriculum.
Skippable; gate (per notes) is a >= 0.5 lower stage-1 text loss at
step 5 000 vs no-stage-0 baseline. If the gate doesn't hold, do not
adopt.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from vista_ocr.data.collate import collate  # noqa: E402
from vista_ocr.data.pdfa import PdfaConfig  # noqa: E402
from vista_ocr.data.preprocess import PreprocessConfig  # noqa: E402
from vista_ocr.data.single_line import SingleLineConfig, iter_single_line_samples  # noqa: E402
from vista_ocr.logging_config import setup_logging  # noqa: E402
from vista_ocr.models.decoder import small_random_decoder  # noqa: E402
from vista_ocr.models.encoder import FCNEncoderWidther  # noqa: E402
from vista_ocr.models.vista_ocr import VistaOCR  # noqa: E402
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid  # noqa: E402
from vista_ocr.tokenizer.tokenizer import VistaTokenizer  # noqa: E402
from vista_ocr.training.callbacks import CheckpointConfig, ValConfig  # noqa: E402
from vista_ocr.training.train_loop import TrainConfig, train  # noqa: E402
from vista_ocr.training.val_helpers import (  # noqa: E402
    make_val_decode_fn,
    make_val_loss_fn,
    pdfa_val_batches,
)

LOG = logging.getLogger("stage0")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shards", nargs="+", required=True, type=Path)
    ap.add_argument("--val-shard", required=True, type=Path)
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--page-h", type=int, default=128,
                    help="Single-line crops; default canvas is small.")
    ap.add_argument("--page-w", type=int, default=1024)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--keep-last", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--decode-n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=args.out / "stage0.log")

    torch.manual_seed(args.seed)
    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(args.spm), grid=grid)

    encoder = FCNEncoderWidther(input_channels=1, dropout=0.5, gradient_checkpointing=True)
    decoder = small_random_decoder(
        vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=4, n_heads=16,
        ffn_dim=4096, max_position_embeddings=4096,
    )
    model = VistaOCR(encoder=encoder, decoder=decoder).cuda()
    LOG.info("Total params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    pre_cfg = PreprocessConfig(
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
    )

    sl_cfg = SingleLineConfig()

    def stage0_stream():
        # Each shard is iterated in turn; line-crops yielded in reading
        # order (top-to-bottom of each page).
        for shard in args.train_shards:
            cfg_p = PdfaConfig(shards=[str(shard)])
            for sample in iter_single_line_samples(cfg_p, sl_cfg):
                yield collate([sample], tokenizer, pre_cfg)

    spatial_ids = tokenizer._spatial_ids

    def val_batches_factory():
        return pdfa_val_batches(args.val_shard, tokenizer, pre_cfg)

    cfg = TrainConfig(
        base_lr=args.lr, warmup_steps=min(500, args.steps // 10),
        total_steps=args.steps,
        micro_batch_size=1, grad_accum_steps=1, log_every=100,
        lambda_text=1.0,                # text-only loss; spatial bbox is trivial
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
        device="cuda", autocast_dtype=torch.bfloat16, gradient_checkpointing=True,
        freeze_decoder=False,
        adam_betas=(0.9, 0.98), adam_eps=1e-6, label_smoothing=0.1,
        min_lr_ratio=args.min_lr_ratio,
        encoder_dropout_max=None,        # mirrors stage-1's choice
        checkpoint=CheckpointConfig(
            out_dir=args.out, save_every=args.ckpt_every, keep_last=args.keep_last,
        ),
        val=ValConfig(every=args.val_every, max_batches=args.val_batches),
        val_batches_factory=val_batches_factory,
        val_loss_fn=make_val_loss_fn(spatial_ids, lambda_text=1.0),
        val_decode_fn=make_val_decode_fn(tokenizer) if args.decode_n > 0 else None,
        val_decode_n=args.decode_n,
    )

    t0 = time.perf_counter()
    history = train(model=model, sample_stream=stage0_stream(),
                    tokenizer=tokenizer, cfg=cfg, max_steps=args.steps)
    elapsed = time.perf_counter() - t0
    if history:
        first = sum(h.loss for h in history[:50]) / max(1, len(history[:50]))
        last = sum(h.loss for h in history[-50:]) / max(1, len(history[-50:]))
        LOG.info("DONE: %d steps, %.1fs (%.3fs/step). Loss first=%.3f last=%.3f delta=%.3f",
                 len(history), elapsed, elapsed / max(1, len(history)),
                 first, last, first - last)


if __name__ == "__main__":
    main()
