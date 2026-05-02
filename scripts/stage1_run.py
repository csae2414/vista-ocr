"""Stage-1 calibration with checkpoint + resume + validation.

Exercises the full long-run plumbing on real PDFA data:

- DataLoader with workers, donut4 decoder, frozen for stage-1a
- Periodic validation on a held-out shard
- Checkpoint every N steps with auto-resume
- ckpt_best.pt on val_loss improvement
- A4 ``min_lr_ratio`` floor on the cosine schedule
- B3 CER/WER reported on the first ``--decode-n`` val batches

Stage-1 has the decoder frozen, so the encoder is the only learner.
B1 dropout schedule is intentionally disabled here (see notes).

Usage on the VM::

    python scripts/stage1_run.py \\
        --train-shards data/raw/pdfa/pdfa-eng-train-{0000,0001,0002}.tar \\
        --val-shard    data/raw/pdfa/pdfa-eng-train-0003.tar \\
        --spm          data/processed/vocab/sp_en_16k.model \\
        --out          checkpoints/stage1 \\
        --steps 2000 --val-every 500 --ckpt-every 500
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

# Avoid cuBLAS workspace contention on long bf16 attention matmuls --
# stabilises a subtle transformers 4.44 + RTX 3090 + bf16 path.
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
from vista_ocr.training.callbacks import CheckpointConfig, ValConfig  # noqa: E402
from vista_ocr.training.train_loop import TrainConfig, train  # noqa: E402
from vista_ocr.training.val_helpers import (  # noqa: E402
    make_val_decode_fn,
    make_val_loss_fn,
    pdfa_val_batches,
)

LOG = logging.getLogger("stage1")


def main() -> None:
    ap = argparse.ArgumentParser()
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
                    choices=("tiny", "small", "medium", "large", "auto"),
                    help="Page resolution preset (default medium = 1100x850, "
                         "fits a 24 GB 3090). 'auto' queries CUDA VRAM. "
                         "Ignored when --page-h/--page-w are set explicitly.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--keep-last", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05,
                    help="A4: cosine-schedule floor as a fraction of base_lr.")
    ap.add_argument("--decode-n", type=int, default=5,
                    help="B3: decode + score CER/WER on first N val batches "
                         "each val call. 0 disables.")
    ap.add_argument("--augment", action="store_true",
                    help="B2: enable train-time bbox-aware augmentation "
                         "(rotation, brightness/contrast, blur, JPEG).")
    ap.add_argument("--sdpa", action="store_true",
                    help="C3: monkey-patch MBartAttention to use SDPA. "
                         "Runs the ship-gate first; aborts on failure.")
    ap.add_argument("--init-decoder-from", default="random",
                    choices=("random", "donut"),
                    help="Decoder weight initialisation. 'random' is the "
                         "from-scratch default; 'donut' loads "
                         "naver-clova-ix/donut-base body weights (paper "
                         "Section 3.2; see vista_ocr.models.donut_init).")
    ap.add_argument("--grad-accum-steps", type=int, default=1,
                    help="Effective batch = micro_batch * accum. Paper "
                         "uses ~11 on A100-80GB; raise to 8 on a 24 GB "
                         "3090 for paper-comparable gradient signal.")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="Disable encoder gradient checkpointing. Faster "
                         "per step but ~3-5x more activation memory; safe "
                         "on 48 GB+ cards.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=args.out / "stage1.log")

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
    encoder = FCNEncoderWidther(input_channels=1, dropout=0.5, gradient_checkpointing=grad_ckpt)
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
            micro_batch_size=1, num_workers=args.num_workers, prefetch_factor=4,
        ),
    )

    spatial_ids = tokenizer._spatial_ids

    def val_batches_factory():
        return pdfa_val_batches(args.val_shard, tokenizer, pre_cfg)

    train_cfg = TrainConfig(
        base_lr=args.lr, warmup_steps=50, total_steps=args.steps,
        micro_batch_size=1, grad_accum_steps=args.grad_accum_steps, log_every=100,
        lambda_text=1.0, target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
        device="cuda", autocast_dtype=torch.bfloat16, gradient_checkpointing=grad_ckpt,
        freeze_decoder=True, adam_betas=(0.9, 0.98), adam_eps=1e-6,
        label_smoothing=0.1,
        # A4: keep a small LR through the cosine tail.
        min_lr_ratio=args.min_lr_ratio,
        # B1: stage-1 has the decoder frozen, so the encoder is the only
        # learner -- starting at p=0 (the schedule) risks early overfit.
        # Notes-spec keeps dropout fixed for stage-1.
        encoder_dropout_max=None,
        checkpoint=CheckpointConfig(
            out_dir=args.out, save_every=args.ckpt_every, keep_last=args.keep_last,
        ),
        val=ValConfig(every=args.val_every, max_batches=args.val_batches),
        val_batches_factory=val_batches_factory,
        val_loss_fn=make_val_loss_fn(spatial_ids, lambda_text=1.0),
        # B3: diagnostic decode + CER/WER. Disabled when --decode-n=0.
        val_decode_fn=make_val_decode_fn(tokenizer) if args.decode_n > 0 else None,
        val_decode_n=args.decode_n,
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
