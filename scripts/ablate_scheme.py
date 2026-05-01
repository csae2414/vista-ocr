"""Ablation: encoding scheme (Original / Segmented / Unified) and quantizer.

Reproduces the structure of paper Table 7. Trains a small model for N
steps under each scheme and compares the converged total loss + the
spatial-token vocab size each scheme requires.

Each scheme requires its own SentencePiece model because the spatial
tokens differ. We train a tiny SPM per scheme on the fly using the same
bootstrap corpus.
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
from vista_ocr.tokenizer.build_spm import train_spm
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import (
    VistaTokenizer,
    list_special_and_spatial_tokens,
)
from vista_ocr.training.train_loop import TrainConfig, train

LOG = logging.getLogger("ablate_scheme")


def _build_tokenizer_for_scheme(
    scheme: str, quantizer_px: int, work_dir: Path, corpus: Path
) -> VistaTokenizer:
    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=quantizer_px, scheme=scheme)
    out_prefix = work_dir / f"sp_{scheme}_q{quantizer_px}"
    if not out_prefix.with_suffix(".model").exists():
        LOG.info("Training SPM for scheme=%s q=%d ...", scheme, quantizer_px)
        train_spm(
            corpus_path=corpus, out_prefix=out_prefix, vocab_size=16000,
            user_symbols=list_special_and_spatial_tokens(grid),
        )
    return VistaTokenizer(spm_model_path=out_prefix.with_suffix(".model"), grid=grid)


def _run(*, scheme: str, quantizer_px: int, shards: list[Path], corpus: Path,
         work_dir: Path, steps: int, page_h: int, page_w: int):
    torch.manual_seed(0)
    tokenizer = _build_tokenizer_for_scheme(scheme, quantizer_px, work_dir, corpus)
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
        lambda_text=0.5, target_h=page_h, target_w=page_w, pad_multiple=32,
        device="cuda", autocast_dtype=torch.bfloat16, gradient_checkpointing=True,
        freeze_decoder=True,
        adam_betas=(0.9, 0.98), adam_eps=1e-6, label_smoothing=0.1,
    )

    t0 = time.perf_counter()
    history = train(model, iter(loader), tokenizer, cfg, max_steps=steps)
    elapsed = time.perf_counter() - t0
    last = sum(h.loss for h in history[-50:]) / max(1, len(history[-50:]))
    return {
        "scheme": scheme, "quantizer": quantizer_px,
        "spatial_tokens": len(tokenizer._spatial_ids),
        "elapsed": elapsed, "loss": last,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--page-h", type=int, default=1100)
    ap.add_argument("--page-w", type=int, default=850)
    args = ap.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(level="INFO")

    configs = [
        ("original", 10),
        ("segmented", 10),
        ("unified", 10),
        ("original", 3),
    ]
    results = []
    for scheme, q in configs:
        LOG.info("=" * 60)
        LOG.info("Scheme=%s quantizer=%dpx", scheme, q)
        r = _run(scheme=scheme, quantizer_px=q, shards=args.shards,
                 corpus=args.corpus, work_dir=args.work_dir,
                 steps=args.steps, page_h=args.page_h, page_w=args.page_w)
        results.append(r)
        LOG.info("scheme=%s q=%d  spatial=%d  loss=%.3f", scheme, q, r["spatial_tokens"], r["loss"])

    LOG.info("=" * 60)
    LOG.info("ENCODING-SCHEME ABLATION (steps=%d each)", args.steps)
    LOG.info(f"{'scheme':<11}{'quant':<8}{'spatial':<10}{'loss':<10}{'wall(s)':<10}")
    for r in results:
        LOG.info(f"{r['scheme']:<11}{r['quantizer']:<8}{r['spatial_tokens']:<10}"
                 f"{r['loss']:<10.3f}{r['elapsed']:<10.1f}")


if __name__ == "__main__":
    main()
