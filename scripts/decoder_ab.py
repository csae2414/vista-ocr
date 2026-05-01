"""Decoder A/B: 12-layer mBART vs 4-layer Donut-style.

The paper claims 150M total. With 12-layer mBART our model is 243.7M;
with 4-layer Donut-style we are at 92.5M. Neither matches; we want
empirical evidence on accuracy + throughput before committing pretraining
GPU-time to one variant. Both runs use the same seed, the same 4 PDFA
shards, bf16 + grad checkpointing + workers=4.
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
from vista_ocr.models.decoder import MBartDecoder, small_random_decoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import VistaTokenizer
from vista_ocr.training.train_loop import TrainConfig, train

LOG = logging.getLogger("ab")


def _run(
    *, name: str, decoder, shards: list[Path], spm: Path, steps: int,
    page_h: int, page_w: int, num_workers: int, seed: int,
):
    torch.manual_seed(seed)
    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(spm), grid=grid)
    encoder = FCNEncoderWidther(input_channels=1, dropout=0.5, gradient_checkpointing=True)
    model = VistaOCR(encoder=encoder, decoder=decoder)

    pre_cfg = PreprocessConfig(target_h=page_h, target_w=page_w, pad_multiple=32)
    loader = make_pdfa_dataloader(
        shards=[str(s) for s in shards],
        tokenizer=tokenizer, pre_cfg=pre_cfg,
        dl_cfg=DataLoaderConfig(micro_batch_size=1, num_workers=num_workers, prefetch_factor=4),
    )

    cfg = TrainConfig(
        base_lr=3e-4, warmup_steps=50, total_steps=steps,
        micro_batch_size=1, grad_accum_steps=1, log_every=200,
        lambda_text=1.0, target_h=page_h, target_w=page_w, pad_multiple=32,
        device="cuda", autocast_dtype=torch.bfloat16, gradient_checkpointing=True,
        freeze_decoder=True, adam_betas=(0.9, 0.98), adam_eps=1e-6,
        label_smoothing=0.1,
    )

    n_params = sum(p.numel() for p in model.parameters())
    LOG.info("[%s] %.1fM params", name, n_params / 1e6)
    t0 = time.perf_counter()
    history = train(model, iter(loader), tokenizer, cfg, max_steps=steps)
    elapsed = time.perf_counter() - t0
    return {
        "name": name,
        "params": n_params,
        "elapsed": elapsed,
        "per_step": elapsed / max(1, len(history)),
        "loss_first": history[0].loss if history else float("nan"),
        "loss_last": history[-1].loss if history else float("nan"),
        "loss_mean_last100": (
            sum(h.loss for h in history[-100:]) / max(1, len(history[-100:]))
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, type=Path)
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--page-h", type=int, default=1100)
    ap.add_argument("--page-w", type=int, default=850)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", choices=("mbart12", "donut4", "both"), default="both")
    args = ap.parse_args()

    setup_logging(level="INFO")

    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(args.spm), grid=grid)
    vocab = tokenizer.vocab_size

    results = []
    if args.variant in ("mbart12", "both"):
        LOG.info("=" * 60)
        LOG.info("Variant A: 12-layer mBART (paper-literal)")
        dec = MBartDecoder.from_pretrained_mbart50(
            vocab_size=vocab, decoder_layers=12, max_position_embeddings=4096,
            load_pretrained_body=False,                 # bench: skip 2GB download
        )
        results.append(_run(
            name="mbart12", decoder=dec, shards=args.shards, spm=args.spm,
            steps=args.steps, page_h=args.page_h, page_w=args.page_w,
            num_workers=args.num_workers, seed=args.seed,
        ))

    if args.variant in ("donut4", "both"):
        LOG.info("=" * 60)
        LOG.info("Variant B: 4-layer Donut-style")
        dec = small_random_decoder(
            vocab_size=vocab, d_model=1024, n_layers=4, n_heads=16, ffn_dim=4096,
            max_position_embeddings=4096,
        )
        results.append(_run(
            name="donut4", decoder=dec, shards=args.shards, spm=args.spm,
            steps=args.steps, page_h=args.page_h, page_w=args.page_w,
            num_workers=args.num_workers, seed=args.seed,
        ))

    LOG.info("=" * 60)
    LOG.info("DECODER A/B SUMMARY")
    LOG.info(f"{'name':<10}{'params':<12}{'wall':<10}{'per_step':<11}{'loss_first':<13}{'loss_last':<13}{'mean_last100':<13}")
    for r in results:
        LOG.info(f"{r['name']:<10}{r['params']/1e6:<12.1f}{r['elapsed']:<10.1f}"
                 f"{r['per_step']:<11.3f}{r['loss_first']:<13.3f}{r['loss_last']:<13.3f}"
                 f"{r['loss_mean_last100']:<13.3f}")


if __name__ == "__main__":
    main()
