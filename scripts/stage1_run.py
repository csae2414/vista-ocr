"""Stage-1 calibration with checkpoint + resume + validation.

Exercises the full long-run plumbing on real PDFA data:

- DataLoader with workers, donut4 decoder, frozen for stage-1a
- Periodic validation on a held-out shard
- Checkpoint every N steps with auto-resume
- ckpt_best.pt on val_loss improvement

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

import torch

from vista_ocr.data.collate import collate
from vista_ocr.data.dataloader import DataLoaderConfig, make_pdfa_dataloader
from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa
from vista_ocr.data.preprocess import PreprocessConfig
from vista_ocr.logging_config import setup_logging
from vista_ocr.models.decoder import small_random_decoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import VistaTokenizer
from vista_ocr.training.callbacks import CheckpointConfig, ValConfig
from vista_ocr.training.losses import combined_loss
from vista_ocr.training.train_loop import TrainConfig, train

LOG = logging.getLogger("stage1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shards", nargs="+", required=True, type=Path)
    ap.add_argument("--val-shard", required=True, type=Path)
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--page-h", type=int, default=1100)
    ap.add_argument("--page-w", type=int, default=850)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--keep-last", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=args.out / "stage1.log")

    torch.manual_seed(args.seed)
    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(args.spm), grid=grid)

    encoder = FCNEncoderWidther(input_channels=1, dropout=0.5, gradient_checkpointing=True)
    decoder = small_random_decoder(
        vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=4, n_heads=16,
        ffn_dim=4096, max_position_embeddings=4096,
    )
    model = VistaOCR(encoder=encoder, decoder=decoder)
    LOG.info("Total params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    pre_cfg = PreprocessConfig(target_h=args.page_h, target_w=args.page_w, pad_multiple=32)

    train_loader = make_pdfa_dataloader(
        shards=[str(p) for p in args.train_shards],
        tokenizer=tokenizer, pre_cfg=pre_cfg,
        dl_cfg=DataLoaderConfig(
            micro_batch_size=1, num_workers=args.num_workers, prefetch_factor=4,
        ),
    )

    spatial_ids = tokenizer._spatial_ids

    def val_batches_factory():
        cfg = PdfaConfig(shards=[str(args.val_shard)])
        for s in iter_pdfa(cfg):
            yield collate([s], tokenizer, pre_cfg)

    def val_loss_fn(model, batch):
        # PyTorch issue #132613 + cuBLAS workspace contention: bf16
        # autocast hits CUBLAS_STATUS_EXECUTION_FAILED in MBart eager
        # self-attention specifically in eval mode + no_grad + autocast.
        # Diagnosed empirically and matches the failure mode reported in
        # the PyTorch issue. Workaround: run val in fp32 (no autocast).
        # Val is infrequent so the slower-but-stable path is fine.
        device = next(model.parameters()).device
        logits = model(
            batch.images.to(device), batch.decoder_input_ids.to(device),
        )
        out = combined_loss(
            logits=logits,
            labels=batch.labels.to(device),
            spatial_token_ids=spatial_ids,
            lambda_text=1.0,
            pad_id=batch.pad_id,
            prompt_mask=batch.prompt_mask.to(device),
        )
        return out.loss

    train_cfg = TrainConfig(
        base_lr=args.lr, warmup_steps=50, total_steps=args.steps,
        micro_batch_size=1, grad_accum_steps=1, log_every=100,
        lambda_text=1.0, target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
        device="cuda", autocast_dtype=torch.bfloat16, gradient_checkpointing=True,
        freeze_decoder=True, adam_betas=(0.9, 0.98), adam_eps=1e-6,
        label_smoothing=0.1,
        checkpoint=CheckpointConfig(
            out_dir=args.out, save_every=args.ckpt_every, keep_last=args.keep_last,
        ),
        val=ValConfig(every=args.val_every, max_batches=args.val_batches),
        val_batches_factory=val_batches_factory,
        val_loss_fn=val_loss_fn,
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
