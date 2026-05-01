"""Ablation: lambda balance between text and location losses.

Paper formula: L = lambda * L_text + (1 - lambda) * L_loc.

The paper does NOT report a value. PLAN_VM lists {0.3, 0.5, 0.7} as the
sweep. This script runs the same number of steps for each lambda and
reports CER + DetEval on a small held-out PDFA slice so we can pick the
best value before launching long pretraining.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from vista_ocr.data.dataloader import DataLoaderConfig, make_pdfa_dataloader
from vista_ocr.data.preprocess import PreprocessConfig
from vista_ocr.logging_config import setup_logging
from vista_ocr.models.decoder import small_random_decoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import VistaTokenizer
from vista_ocr.training.train_loop import TrainConfig, train

LOG = logging.getLogger("ablate_lambda")


def _run(*, lam: float, shards: list[Path], spm: Path, steps: int, page_h: int, page_w: int):
    torch.manual_seed(0)
    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(spm), grid=grid)
    encoder = FCNEncoderWidther(input_channels=1, dropout=0.5, gradient_checkpointing=True)
    decoder = small_random_decoder(
        vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=4, n_heads=16,
        ffn_dim=4096, max_position_embeddings=4096,
    )
    model = VistaOCR(encoder=encoder, decoder=decoder)

    pre_cfg = PreprocessConfig(target_h=page_h, target_w=page_w, pad_multiple=32)
    loader = make_pdfa_dataloader(
        shards=[str(s) for s in shards],
        tokenizer=tokenizer, pre_cfg=pre_cfg,
        dl_cfg=DataLoaderConfig(micro_batch_size=1, num_workers=4, prefetch_factor=4),
    )

    cfg = TrainConfig(
        base_lr=3e-4, warmup_steps=50, total_steps=steps,
        micro_batch_size=1, grad_accum_steps=1, log_every=100,
        lambda_text=lam, target_h=page_h, target_w=page_w, pad_multiple=32,
        device="cuda", autocast_dtype=torch.bfloat16, gradient_checkpointing=True,
        freeze_decoder=False,                # full training to test loss balance
        adam_betas=(0.9, 0.98), adam_eps=1e-6,
        label_smoothing=0.1,
    )

    t0 = time.perf_counter()
    history = train(model, iter(loader), tokenizer, cfg, max_steps=steps)
    elapsed = time.perf_counter() - t0
    last_text = sum(h.loss_text for h in history[-50:]) / max(1, len(history[-50:]))
    last_loc = sum(h.loss_loc for h in history[-50:]) / max(1, len(history[-50:]))
    last_total = sum(h.loss for h in history[-50:]) / max(1, len(history[-50:]))
    return {
        "lambda": lam, "elapsed": elapsed, "steps": len(history),
        "loss_total": last_total, "loss_text": last_text, "loss_loc": last_loc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, type=Path)
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--page-h", type=int, default=1100)
    ap.add_argument("--page-w", type=int, default=850)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    args = ap.parse_args()

    setup_logging(level="INFO")

    results = []
    for lam in args.lambdas:
        LOG.info("=" * 60)
        LOG.info("Lambda = %.2f", lam)
        r = _run(lam=lam, shards=args.shards, spm=args.spm,
                 steps=args.steps, page_h=args.page_h, page_w=args.page_w)
        results.append(r)
        LOG.info("lambda=%.2f total=%.3f text=%.3f loc=%.3f", lam, r["loss_total"], r["loss_text"], r["loss_loc"])

    LOG.info("=" * 60)
    LOG.info("LAMBDA ABLATION SUMMARY (steps=%d each)", args.steps)
    LOG.info(f"{'lambda':<8}{'total':<10}{'text':<10}{'loc':<10}{'wall(s)':<10}")
    for r in results:
        LOG.info(f"{r['lambda']:<8.2f}{r['loss_total']:<10.3f}{r['loss_text']:<10.3f}"
                 f"{r['loss_loc']:<10.3f}{r['elapsed']:<10.1f}")


if __name__ == "__main__":
    main()
