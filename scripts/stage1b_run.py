"""Stage-1b: unfrozen OCR-only calibration (Phase I).

Paper §3.6.1 splits stage 1 into two sub-stages:

* Stage 1a (frozen-decoder text-only) -- our existing
  ``stage1_run.py``. Encoder calibrates against a frozen mBART
  decoder.
* **Stage 1b (this script)**: unfrozen-all text-only. Decoder
  warms up to the encoder's features without yet being asked to
  handle layout. Bridges the curriculum gap between stage 1
  (frozen) and stage 2 (unfrozen + multimodal).

Differences from ``stage1_run.py``:

* ``freeze_decoder=False`` (the whole point of this stage).
* ``lambda_text=1.0`` (still text-only; layout is stage 2's job).
* ``--lr`` defaults to ``5e-5`` (matching stage 2). Aggressive
  stage-1-style ``3e-4`` would damage the warm start.
* ``--init-from`` REQUIRED -- this stage only makes sense as a
  transition from a stage-1 ckpt; running fresh wouldn't help.

Usage on the VM::

    python scripts/stage1b_run.py \\
        --train-shards data/raw/pdfa/pdfa-eng-train-{0000,0001,0002}.tar \\
        --val-shard    data/raw/pdfa/pdfa-eng-train-0118.tar \\
        --spm          data/processed/vocab/sp_en_16k.model \\
        --init-from    checkpoints/stage1/ckpt_final.pt \\
        --out          checkpoints/stage1b \\
        --steps 10000 --val-every 500 --ckpt-every 500
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from vista_ocr.data.augment import AugmentConfig  # noqa: E402
from vista_ocr.data.dataloader import (  # noqa: E402
    DataLoaderConfig,
    make_mixed_pdfa_idl_loader,
    make_pdfa_dataloader,
)
from vista_ocr.data.preprocess import PreprocessConfig  # noqa: E402
from vista_ocr.data.split import assert_not_test_shard  # noqa: E402
from vista_ocr.logging_config import setup_logging  # noqa: E402
from vista_ocr.models.decoder import small_random_decoder  # noqa: E402
from vista_ocr.models.encoder import FCNEncoderWidther  # noqa: E402
from vista_ocr.models.vista_ocr import VistaOCR  # noqa: E402
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid  # noqa: E402
from vista_ocr.tokenizer.tokenizer import VistaTokenizer  # noqa: E402
from vista_ocr.training.callbacks import (  # noqa: E402
    CheckpointConfig,
    EarlyStopConfig,
    ValConfig,
    load_checkpoint,
)
from vista_ocr.training.train_loop import TrainConfig, train  # noqa: E402
from vista_ocr.training.val_helpers import (  # noqa: E402
    make_val_decode_fn,
    make_val_loss_fn,
    pdfa_val_batches,
)

LOG = logging.getLogger("stage1b")


def _build_early_stop(args: argparse.Namespace) -> EarlyStopConfig | None:
    if not getattr(args, "early_stop", False):
        return None
    return EarlyStopConfig(
        enabled=True,
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        warmup_vals=args.early_stop_warmup,
        metric=args.select_on,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shards", nargs="+", required=True, type=Path)
    ap.add_argument("--idl-shards", nargs="+", type=Path, default=None,
                    help="Phase H: optional PDFA+IDL mix in stage 1b too.")
    ap.add_argument("--idl-weight", type=float, default=0.6)
    ap.add_argument("--val-shard", required=True, type=Path)
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--init-from", required=True, type=Path,
                    help="Stage-1 checkpoint to warm-start from. Required: "
                         "stage 1b only makes sense as a transition step.")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--page-h", type=int, default=None)
    ap.add_argument("--page-w", type=int, default=None)
    ap.add_argument("--page-preset", default="medium",
                    choices=("tiny", "small", "medium", "large", "paper", "auto"))
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--keep-last", type=int, default=3)
    # Stage 1b LR is lower than stage 1 (the decoder is unfrozen and
    # warmed-from-stage-1; aggressive LR would damage the warm start).
    # Matches stage 2's default.
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--decode-n", type=int, default=5)
    ap.add_argument("--select-on", default="val_word_f1",
                    choices=("val_loss", "val_word_f1"))
    ap.add_argument("--decode-n-best", type=int, default=256)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--sdpa", action="store_true")
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--compile", action="store_true", dest="compile_model")
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--early-stop-patience", type=int, default=15,
                    help="Stage 1b val_loss converges faster than stage 1's "
                         "(decoder is now learning); 15 is plenty.")
    ap.add_argument("--early-stop-min-delta", type=float, default=0.005)
    ap.add_argument("--early-stop-warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert_not_test_shard(args.val_shard)
    for s in args.train_shards:
        assert_not_test_shard(s)

    args.out.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=args.out / "stage1b.log")

    from vista_ocr.training.resolution import resolve as resolve_resolution
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
    model = VistaOCR(encoder=encoder, decoder=decoder).cuda()

    if args.init_from and args.init_from.exists():
        LOG.info("Initialising from stage-1 checkpoint %s", args.init_from)
        load_checkpoint(args.init_from, model=model, optimizer=None,
                        map_location="cuda", strict=False, restore_rng=False)
    else:
        raise SystemExit(
            f"--init-from missing: {args.init_from}. Stage 1b is a "
            "transition stage and requires a stage-1 ckpt to warm-start from."
        )

    LOG.info("Total params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    aug_cfg = AugmentConfig(enabled=True) if args.augment else None
    pre_cfg = PreprocessConfig(
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
        augment=aug_cfg,
    )
    if aug_cfg is not None:
        LOG.info("B2: train-time augmentation enabled")

    dl_cfg = DataLoaderConfig(
        micro_batch_size=1, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    if args.idl_shards:
        idl_w = args.idl_weight
        pdfa_w = 1.0 - idl_w
        LOG.info("Phase H: PDFA+IDL mix (%.2f / %.2f)", pdfa_w, idl_w)
        train_loader = make_mixed_pdfa_idl_loader(
            pdfa_shards=[str(p) for p in args.train_shards],
            idl_shards=[str(p) for p in args.idl_shards],
            tokenizer=tokenizer, pre_cfg=pre_cfg, dl_cfg=dl_cfg,
            pdfa_weight=pdfa_w, idl_weight=idl_w,
        )
    else:
        train_loader = make_pdfa_dataloader(
            shards=[str(p) for p in args.train_shards],
            tokenizer=tokenizer, pre_cfg=pre_cfg, dl_cfg=dl_cfg,
        )

    spatial_ids = tokenizer._spatial_ids

    def val_batches_factory():
        return pdfa_val_batches(args.val_shard, tokenizer, pre_cfg)

    train_cfg = TrainConfig(
        base_lr=args.lr, warmup_steps=500, total_steps=args.steps,
        micro_batch_size=1, grad_accum_steps=args.grad_accum_steps, log_every=100,
        # Stage 1b is text-only -- layout (lambda_text < 1.0) is stage 2's job.
        lambda_text=1.0,
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
        device="cuda", autocast_dtype=torch.bfloat16, gradient_checkpointing=grad_ckpt,
        compile_model=args.compile_model,
        # The whole point of stage 1b: decoder unfrozen.
        freeze_decoder=False,
        adam_betas=(0.9, 0.98), adam_eps=1e-6, label_smoothing=0.1,
        min_lr_ratio=args.min_lr_ratio,
        # B1 dropout schedule: stage 1b sees the decoder unfreeze; the
        # encoder dropout schedule is the same as stage 2 (decay from 0).
        encoder_dropout_max=0.5,
        dropout_T=5e4,
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
        early_stop=_build_early_stop(args),
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


if __name__ == "__main__":
    main()
