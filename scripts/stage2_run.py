"""Stage-2 multimodal pretraining (text + interleaved layout).

Resumes from a stage-1 checkpoint, unfreezes the decoder, and trains on
PDFA + IDL + synthetic data with the ``ocr_layout`` task only (text and
spatial tokens interleaved per paper Table 7 "Original" scheme).

Differs from stage-1:

- ``freeze_decoder=False`` (full model trains)
- ``lambda_text=0.5`` (paper-default; balances text and location loss)
- LR drops to 5e-5 (post-calibration)
- A4 ``min_lr_ratio`` floor on the cosine schedule
- B1 DANIEL exponential dropout schedule (encoder learns more freely
  early, full regularisation late)
- B3 CER/WER reported on the first ``--decode-n`` val batches
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
from vista_ocr.data.dataloader import DataLoaderConfig, make_pdfa_dataloader  # noqa: E402
from vista_ocr.data.preprocess import PreprocessConfig  # noqa: E402
from vista_ocr.logging_config import setup_logging  # noqa: E402
from vista_ocr.models.decoder import small_random_decoder  # noqa: E402
from vista_ocr.models.encoder import FCNEncoderWidther  # noqa: E402
from vista_ocr.models.vista_ocr import VistaOCR  # noqa: E402
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid  # noqa: E402
from vista_ocr.tokenizer.tokenizer import VistaTokenizer  # noqa: E402
from vista_ocr.training.callbacks import (  # noqa: E402
    CheckpointConfig,
    ValConfig,
    load_checkpoint,
)
from vista_ocr.training.train_loop import TrainConfig, train  # noqa: E402
from vista_ocr.training.val_helpers import (  # noqa: E402
    make_val_decode_fn,
    make_val_loss_fn,
    pdfa_val_batches,
)

LOG = logging.getLogger("stage2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shards", nargs="+", required=True, type=Path)
    ap.add_argument("--val-shard", required=True, type=Path)
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--init-from", type=Path, default=None,
                    help="stage-1 checkpoint to load encoder/decoder weights from")
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--page-h", type=int, default=None,
                    help="Page canvas height in px. Overrides --page-preset.")
    ap.add_argument("--page-w", type=int, default=None,
                    help="Page canvas width in px. Overrides --page-preset.")
    ap.add_argument("--page-preset", default="medium",
                    choices=("tiny", "small", "medium", "large", "auto"),
                    help="Page resolution preset. 'auto' queries CUDA VRAM. "
                         "Default 'medium' = 1100x850 (24 GB 3090).")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=4,
                    help="Per-worker prefetch buffer.")
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lambda-text", type=float, default=0.5)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05,
                    help="A4: cosine-schedule floor as a fraction of base_lr.")
    ap.add_argument("--encoder-dropout-max", type=float, default=0.5,
                    help="B1: peak DANIEL exponential dropout. None disables.")
    ap.add_argument("--dropout-T", type=float, default=5e4,
                    help="B1: time-constant of the dropout schedule.")
    ap.add_argument("--decode-n", type=int, default=5,
                    help="B3: decode + score CER/WER on first N val batches.")
    ap.add_argument("--augment", action="store_true",
                    help="B2: enable train-time bbox-aware augmentation.")
    ap.add_argument("--sdpa", action="store_true",
                    help="C3: monkey-patch MBartAttention to use SDPA. "
                         "Runs the ship-gate first; aborts on failure.")
    ap.add_argument("--grad-accum-steps", type=int, default=1,
                    help="Effective batch = micro_batch * accum. 8 is "
                         "paper-comparable on a 24 GB 3090.")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="Disable encoder gradient checkpointing.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=args.out / "stage2.log")

    from vista_ocr.training.resolution import resolve as resolve_resolution
    if args.page_h is None or args.page_w is None:
        res = resolve_resolution(args.page_preset)
        args.page_h = args.page_h or res.height
        args.page_w = args.page_w or res.width
    LOG.info("Page canvas: %d x %d", args.page_h, args.page_w)

    if args.sdpa:
        from vista_ocr.models.sdpa_patch import enable_with_ship_gate
        enable_with_ship_gate(manifest_dir=args.out)

    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(args.spm), grid=grid)

    grad_ckpt = not args.no_grad_ckpt
    encoder = FCNEncoderWidther(input_channels=1, dropout=0.5, gradient_checkpointing=grad_ckpt)
    decoder = small_random_decoder(
        vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=4, n_heads=16,
        ffn_dim=4096, max_position_embeddings=4096,
    )
    model = VistaOCR(encoder=encoder, decoder=decoder).cuda()

    if args.init_from and args.init_from.exists():
        LOG.info("Initialising from stage-1 checkpoint %s", args.init_from)
        load_checkpoint(args.init_from, model=model, optimizer=None,
                        map_location="cuda", strict=False, restore_rng=False)

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

    cfg = TrainConfig(
        base_lr=args.lr, warmup_steps=2000, total_steps=args.steps,
        micro_batch_size=1, grad_accum_steps=args.grad_accum_steps, log_every=200,
        lambda_text=args.lambda_text,
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
        device="cuda", autocast_dtype=torch.bfloat16, gradient_checkpointing=grad_ckpt,
        freeze_decoder=False,
        adam_betas=(0.9, 0.98), adam_eps=1e-6, label_smoothing=0.1,
        # A4
        min_lr_ratio=args.min_lr_ratio,
        # B1
        encoder_dropout_max=args.encoder_dropout_max,
        dropout_T=args.dropout_T,
        checkpoint=CheckpointConfig(
            out_dir=args.out, save_every=args.ckpt_every, keep_last=3,
        ),
        val=ValConfig(every=args.val_every, max_batches=args.val_batches),
        val_batches_factory=val_batches_factory,
        val_loss_fn=make_val_loss_fn(spatial_ids, lambda_text=args.lambda_text),
        # B3
        val_decode_fn=make_val_decode_fn(tokenizer) if args.decode_n > 0 else None,
        val_decode_n=args.decode_n,
    )

    LOG.info("Stage-2 multimodal pretraining: %d steps, lambda=%.2f, lr=%.2e",
             args.steps, args.lambda_text, args.lr)
    t0 = time.perf_counter()
    history = train(model, sample_stream=iter(train_loader), tokenizer=tokenizer,
                    cfg=cfg, max_steps=args.steps)
    elapsed = time.perf_counter() - t0
    if history:
        first = sum(h.loss for h in history[:50]) / max(1, len(history[:50]))
        last = sum(h.loss for h in history[-50:]) / max(1, len(history[-50:]))
        LOG.info("DONE: %d steps in %.1fs (%.3fs/step). loss first=%.3f last=%.3f delta=%.3f",
                 len(history), elapsed, elapsed / max(1, len(history)),
                 first, last, first - last)


if __name__ == "__main__":
    main()
