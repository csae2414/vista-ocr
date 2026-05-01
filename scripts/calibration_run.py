"""Stage-1a calibration: 300 steps on real PDFA, frozen decoder.

Sanity-check run before any long pretraining (PLAN_VM Phase 0). Exits
non-zero if loss does not decrease over the last 100 steps.
"""
from __future__ import annotations

import argparse
import logging
import statistics
import time
from pathlib import Path

import torch

from vista_ocr.data.dataloader import DataLoaderConfig, make_pdfa_dataloader
from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa
from vista_ocr.data.preprocess import PreprocessConfig
from vista_ocr.logging_config import setup_logging
from vista_ocr.models.decoder import MBartDecoder, small_random_decoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import VistaTokenizer
from vista_ocr.training.train_loop import StepStats, TrainConfig, train

LOG = logging.getLogger("calibration")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, type=Path)
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--micro-bs", type=int, default=1)
    ap.add_argument("--page-h", type=int, default=2200)
    ap.add_argument("--page-w", type=int, default=1700)
    ap.add_argument("--lr", type=float, default=3e-4)        # stage-1a frozen-decoder LR
    ap.add_argument("--use-mbart", action="store_true",
                    help="Real 12-layer mbart-large-50 decoder (downloads ~2 GB)")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader worker processes for PDFA shards. 0 = inline (slow).")
    ap.add_argument("--log-file", type=Path, default=Path("logs/calibration.log"))
    args = ap.parse_args()

    setup_logging(level="INFO", log_file=args.log_file)
    LOG.info("PyTorch %s | CUDA %s", torch.__version__, torch.cuda.is_available())

    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(args.spm), grid=grid)
    LOG.info("Tokenizer: vocab=%d, spatial=%d", tokenizer.vocab_size, len(tokenizer._spatial_ids))

    encoder = FCNEncoderWidther(input_channels=1, dropout=0.5, gradient_checkpointing=True)
    if args.use_mbart:
        LOG.info("Loading mbart-large-50 (12-layer)...")
        decoder = MBartDecoder.from_pretrained_mbart50(
            vocab_size=tokenizer.vocab_size,
            decoder_layers=12,
            max_position_embeddings=4096,
            load_pretrained_body=True,
        )
    else:
        LOG.info("Using small random decoder (--use-mbart disabled)")
        decoder = small_random_decoder(
            vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=4, n_heads=8, ffn_dim=2048,
            max_position_embeddings=4096,
        )
    model = VistaOCR(encoder=encoder, decoder=decoder)
    LOG.info("Total params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    pre_cfg = PreprocessConfig(
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32
    )
    if args.num_workers > 0:
        LOG.info("Using DataLoader with %d workers + prefetch=4", args.num_workers)
        loader = make_pdfa_dataloader(
            shards=[str(args.shard)],
            tokenizer=tokenizer,
            pre_cfg=pre_cfg,
            dl_cfg=DataLoaderConfig(
                micro_batch_size=args.micro_bs,
                num_workers=args.num_workers,
                prefetch_factor=4,
            ),
        )
        sample_stream = iter(loader)
    else:
        LOG.info("Using inline (single-process) PDFA iterator")
        sample_stream = iter_pdfa(PdfaConfig(shards=[str(args.shard)]))

    train_cfg = TrainConfig(
        base_lr=args.lr,
        warmup_steps=20,
        total_steps=args.steps,
        micro_batch_size=args.micro_bs,
        grad_accum_steps=1,
        log_every=10,
        lambda_text=1.0,                     # OCR-only output -> all weight on text
        target_h=args.page_h,
        target_w=args.page_w,
        pad_multiple=32,
        device="cuda",
        autocast_dtype=torch.bfloat16,
        gradient_checkpointing=True,
        freeze_decoder=True,
        adam_betas=(0.9, 0.98),
        adam_eps=1e-6,
        label_smoothing=0.1,
    )

    history: list[StepStats] = []
    t0 = time.perf_counter()
    history = train(
        model=model,
        sample_stream=sample_stream,
        tokenizer=tokenizer,
        cfg=train_cfg,
        max_steps=args.steps,
    )
    elapsed = time.perf_counter() - t0

    if not history:
        LOG.error("No optimization steps were taken; aborting")
        raise SystemExit(2)

    early = statistics.mean(h.loss for h in history[: max(1, len(history) // 10)])
    late = statistics.mean(h.loss for h in history[-max(1, len(history) // 10):])
    LOG.info(
        "DONE: %d steps in %.1fs (%.2fs/step). Loss early=%.3f late=%.3f delta=%.3f",
        len(history), elapsed, elapsed / max(1, len(history)),
        early, late, early - late,
    )
    if late > early:
        LOG.warning("Loss did not decrease end-to-end. Check data + masking.")
        raise SystemExit(3)
    LOG.info("OK -- calibration smoke test passed")


if __name__ == "__main__":
    main()
