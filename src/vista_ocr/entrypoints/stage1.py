"""``vista-ocr stage 1`` -- calibration stage with a frozen decoder.

Mirrors ``scripts/stage1_run.py`` for the CLI. The legacy script
remains the operator-facing entry point (``python scripts/stage1_run.py``);
this module is the importable Python entry that lets the CLI
dispatcher reach the same behaviour. The contract that prevents
drift between the two is in ``tests/test_stage_equivalence.py``:
flag-set parity, TrainConfig snapshot equality, and model
state_dict() key-set equality on a fixed minimal arglist.

Defers ``import torch`` and ``os.environ.setdefault`` into ``run()``
so that ``vista-ocr --help`` (which lazy-imports this module via the
dispatcher) does not pay the torch-import tax.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="vista-ocr stage1",
        description="Stage-1 calibration (frozen decoder, encoder-only training).",
        add_help=add_help,
    )
    ap.add_argument("--train-shards", nargs="+", required=True, type=Path)
    ap.add_argument("--val-shard", required=True, type=Path)
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--page-h", type=int, default=None,
                    help="Page canvas height in px. Overrides --page-preset.")
    ap.add_argument("--page-w", type=int, default=None,
                    help="Page canvas width in px. Overrides --page-preset.")
    ap.add_argument("--page-preset", default="medium",
                    choices=("tiny", "small", "medium", "large", "paper", "auto"),
                    help="Page resolution preset (default medium = 1100x850, "
                         "fits a 24 GB 3090). 'auto' queries CUDA VRAM. "
                         "Ignored when --page-h/--page-w are set explicitly.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=4,
                    help="Per-worker prefetch buffer.")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--keep-last", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05,
                    help="A4: cosine-schedule floor as a fraction of base_lr.")
    ap.add_argument("--decode-n", type=int, default=5,
                    help="B3: decode + score CER/WER on first N val batches.")
    ap.add_argument("--select-on", default="val_word_f1",
                    choices=("val_loss", "val_word_f1"))
    ap.add_argument("--decode-n-best", type=int, default=256,
                    help="DS-fix Phase 2: second-pass eval on ckpt_best "
                         "candidates with this many batches. 0 disables.")
    ap.add_argument("--augment", action="store_true",
                    help="B2: enable train-time bbox-aware augmentation.")
    ap.add_argument("--sdpa", action="store_true",
                    help="C3: monkey-patch MBartAttention to use SDPA.")
    ap.add_argument("--init-decoder-from", default="random",
                    choices=("random", "donut"))
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--compile", action="store_true", dest="compile_model")
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--early-stop-patience", type=int, default=20)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.005)
    ap.add_argument("--early-stop-warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def run(args: argparse.Namespace) -> int:
    import logging
    import os
    import time

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch

    from vista_ocr.data.augment import AugmentConfig
    from vista_ocr.data.dataloader import DataLoaderConfig, make_pdfa_dataloader
    from vista_ocr.data.preprocess import PreprocessConfig
    from vista_ocr.data.split import assert_not_test_shard
    from vista_ocr.logging_config import setup_logging
    from vista_ocr.models.decoder import small_random_decoder
    from vista_ocr.models.encoder import FCNEncoderWidther
    from vista_ocr.models.vista_ocr import VistaOCR
    from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
    from vista_ocr.tokenizer.tokenizer import VistaTokenizer
    from vista_ocr.training.callbacks import (
        CheckpointConfig,
        EarlyStopConfig,
        ValConfig,
    )
    from vista_ocr.training.train_loop import TrainConfig, train
    from vista_ocr.training.val_helpers import (
        make_val_decode_fn,
        make_val_loss_fn,
        pdfa_val_batches,
    )
    from vista_ocr.training.resolution import resolve as resolve_resolution

    LOG = logging.getLogger("stage1")

    assert_not_test_shard(args.val_shard)
    for s in args.train_shards:
        assert_not_test_shard(s)

    args.out.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=args.out / "stage1.log")

    if args.page_h is None or args.page_w is None:
        res = resolve_resolution(args.page_preset)
        args.page_h = args.page_h or res.height
        args.page_w = args.page_w or res.width
    LOG.info("Page canvas: %d x %d", args.page_h, args.page_w)

    if args.sdpa:
        from vista_ocr.models.sdpa_patch import enable_with_ship_gate
        enable_with_ship_gate(manifest_dir=args.out)

    torch.manual_seed(args.seed)
    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(args.spm), grid=grid)

    grad_ckpt = not args.no_grad_ckpt
    encoder = FCNEncoderWidther(
        input_channels=1, dropout=0.5, gradient_checkpointing=grad_ckpt,
    )
    decoder = small_random_decoder(
        vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=4, n_heads=16,
        ffn_dim=4096, max_position_embeddings=4096,
    )
    if args.init_decoder_from == "donut":
        from vista_ocr.models.donut_init import init_decoder_from_donut
        init_decoder_from_donut(decoder)
    model = VistaOCR(encoder=encoder, decoder=decoder)
    LOG.info("Total params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    aug_cfg = AugmentConfig(enabled=True) if args.augment else None
    pre_cfg = PreprocessConfig(
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
        augment=aug_cfg,
    )
    if aug_cfg is not None:
        LOG.info("B2: train-time augmentation enabled")

    train_loader = make_pdfa_dataloader(
        shards=[str(p) for p in args.train_shards],
        tokenizer=tokenizer, pre_cfg=pre_cfg,
        dl_cfg=DataLoaderConfig(
            micro_batch_size=1, num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        ),
    )

    spatial_ids = tokenizer._spatial_ids

    def val_batches_factory():
        return pdfa_val_batches(args.val_shard, tokenizer, pre_cfg)

    early_stop = (
        EarlyStopConfig(
            enabled=True,
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
            warmup_vals=args.early_stop_warmup,
            metric=args.select_on,
        )
        if args.early_stop else None
    )

    train_cfg = TrainConfig(
        base_lr=args.lr, warmup_steps=50, total_steps=args.steps,
        micro_batch_size=1, grad_accum_steps=args.grad_accum_steps, log_every=100,
        lambda_text=1.0, target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
        device="cuda", autocast_dtype=torch.bfloat16, gradient_checkpointing=grad_ckpt,
        compile_model=args.compile_model,
        freeze_decoder=True, adam_betas=(0.9, 0.98), adam_eps=1e-6,
        label_smoothing=0.1,
        min_lr_ratio=args.min_lr_ratio,
        encoder_dropout_max=None,
        checkpoint=CheckpointConfig(
            out_dir=args.out, save_every=args.ckpt_every,
            keep_last=args.keep_last, select_on=args.select_on,
        ),
        val=ValConfig(every=args.val_every, max_batches=args.val_batches),
        val_batches_factory=val_batches_factory,
        val_loss_fn=make_val_loss_fn(spatial_ids, lambda_text=1.0),
        val_decode_fn=make_val_decode_fn(tokenizer) if args.decode_n > 0 else None,
        val_decode_n=args.decode_n,
        val_decode_n_best=args.decode_n_best,
        early_stop=early_stop,
    )

    t0 = time.perf_counter()
    history = train(
        model=model, sample_stream=iter(train_loader),
        tokenizer=tokenizer, cfg=train_cfg, max_steps=args.steps,
    )
    elapsed = time.perf_counter() - t0
    if history:
        first = sum(h.loss for h in history[:50]) / max(1, len(history[:50]))
        last = sum(h.loss for h in history[-50:]) / max(1, len(history[-50:]))
        LOG.info("DONE: %d steps, %.1fs (%.3fs/step). Loss first=%.3f last=%.3f delta=%.3f",
                 len(history), elapsed, elapsed / max(1, len(history)),
                 first, last, first - last)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
