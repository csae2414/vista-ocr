"""Decoder-layer A/B at scale (C1 in the improvements roadmap).

Trains the same configuration with three decoder depths -- 4, 6, 8
layers -- for the same number of steps and reports total params /
val loss / val CER / wall time per variant.

Designed to run AFTER the SPM / dropout / augmentation choices are
settled; switching decoder depth invalidates the resulting checkpoint
so this is genuinely sequential. Per the notes, the gate is:

  spread of final losses < 0.3  -> stay with the current donut4

Cost estimate: ~6 GPU-h on an L40S for 10K steps × 3 variants.

Example::

    python scripts/ablate_decoder_layers.py \\
        --train-shards data/raw/pdfa/pdfa-eng-train-{0000,0001,0002}.tar \\
        --val-shard    data/raw/pdfa/pdfa-eng-train-0003.tar \\
        --spm          data/processed/vocab/sp_en_16k.model \\
        --steps 10000 \\
        --layers 4 6 8
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from vista_ocr.data.dataloader import DataLoaderConfig, make_pdfa_dataloader  # noqa: E402
from vista_ocr.data.preprocess import PreprocessConfig  # noqa: E402
from vista_ocr.logging_config import setup_logging  # noqa: E402
from vista_ocr.models.decoder import small_random_decoder  # noqa: E402
from vista_ocr.models.encoder import FCNEncoderWidther  # noqa: E402
from vista_ocr.models.vista_ocr import VistaOCR  # noqa: E402
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid  # noqa: E402
from vista_ocr.tokenizer.tokenizer import VistaTokenizer  # noqa: E402
from vista_ocr.training.callbacks import ValConfig  # noqa: E402
from vista_ocr.training.train_loop import TrainConfig, train  # noqa: E402
from vista_ocr.training.val_helpers import (  # noqa: E402
    make_val_decode_fn,
    make_val_loss_fn,
    pdfa_val_batches,
)

LOG = logging.getLogger("ablate_layers")


def _run(
    *,
    n_layers: int,
    train_shards: list[Path],
    val_shard: Path,
    tokenizer: VistaTokenizer,
    pre_cfg: PreprocessConfig,
    steps: int,
    page_h: int,
    page_w: int,
    num_workers: int,
    lr: float,
    lambda_text: float,
) -> dict:
    """Train one variant and return a metrics summary."""
    torch.manual_seed(0)
    encoder = FCNEncoderWidther(
        input_channels=1, dropout=0.5, gradient_checkpointing=True,
    )
    decoder = small_random_decoder(
        vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=n_layers,
        n_heads=16, ffn_dim=4096, max_position_embeddings=4096,
    )
    model = VistaOCR(encoder=encoder, decoder=decoder).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    LOG.info("variant n_layers=%d : %.1fM params", n_layers, n_params / 1e6)

    loader = make_pdfa_dataloader(
        shards=[str(p) for p in train_shards],
        tokenizer=tokenizer, pre_cfg=pre_cfg,
        dl_cfg=DataLoaderConfig(
            micro_batch_size=1, num_workers=num_workers, prefetch_factor=4,
        ),
    )
    spatial_ids = tokenizer._spatial_ids

    def vbf():
        return pdfa_val_batches(val_shard, tokenizer, pre_cfg)

    cfg = TrainConfig(
        base_lr=lr, warmup_steps=min(2000, steps // 10),
        total_steps=steps, log_every=200,
        lambda_text=lambda_text,
        target_h=page_h, target_w=page_w, pad_multiple=32,
        device="cuda", autocast_dtype=torch.bfloat16,
        gradient_checkpointing=True,
        adam_betas=(0.9, 0.98), adam_eps=1e-6, label_smoothing=0.1,
        min_lr_ratio=0.05,
        encoder_dropout_max=0.5, dropout_T=5e4,
        val=ValConfig(every=max(steps // 5, 500), max_batches=20),
        val_batches_factory=vbf,
        val_loss_fn=make_val_loss_fn(spatial_ids, lambda_text=lambda_text),
        val_decode_fn=make_val_decode_fn(tokenizer),
        val_decode_n=5,
    )

    t0 = time.perf_counter()
    history = train(model, sample_stream=iter(loader),
                    tokenizer=tokenizer, cfg=cfg, max_steps=steps)
    elapsed = time.perf_counter() - t0

    if not history:
        return {"layers": n_layers, "params": n_params, "elapsed": elapsed,
                "loss_first": float("nan"), "loss_last": float("nan")}

    first = sum(h.loss for h in history[:50]) / max(1, len(history[:50]))
    last = sum(h.loss for h in history[-50:]) / max(1, len(history[-50:]))
    return {
        "layers": n_layers, "params": n_params, "elapsed": elapsed,
        "loss_first": first, "loss_last": last,
        "delta": first - last,
        "per_step": elapsed / max(1, len(history)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shards", nargs="+", required=True, type=Path)
    ap.add_argument("--val-shard", required=True, type=Path)
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--layers", nargs="+", type=int, default=[4, 6, 8])
    ap.add_argument("--page-h", type=int, default=1100)
    ap.add_argument("--page-w", type=int, default=850)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lambda-text", type=float, default=0.5)
    ap.add_argument("--gate-loss-spread", type=float, default=0.3,
                    help="Pass-gate: if max-min final loss across variants "
                         "is below this, declare 'no winner' and keep "
                         "donut4 as default.")
    args = ap.parse_args()

    setup_logging(level="INFO")
    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(args.spm), grid=grid)
    pre_cfg = PreprocessConfig(
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
    )

    results: list[dict] = []
    for n in args.layers:
        LOG.info("=" * 60)
        LOG.info("Variant: %d layers", n)
        results.append(_run(
            n_layers=n, train_shards=args.train_shards, val_shard=args.val_shard,
            tokenizer=tokenizer, pre_cfg=pre_cfg, steps=args.steps,
            page_h=args.page_h, page_w=args.page_w, num_workers=args.num_workers,
            lr=args.lr, lambda_text=args.lambda_text,
        ))

    LOG.info("=" * 60)
    LOG.info("DECODER-LAYER A/B SUMMARY (steps=%d each)", args.steps)
    LOG.info(f"{'layers':<8}{'params(M)':<12}{'wall(s)':<10}{'per_step(s)':<13}{'first':<10}{'last':<10}{'delta':<10}")
    for r in results:
        LOG.info(
            f"{r['layers']:<8}{r['params'] / 1e6:<12.1f}{r['elapsed']:<10.1f}"
            f"{r.get('per_step', 0):<13.3f}{r['loss_first']:<10.3f}"
            f"{r['loss_last']:<10.3f}{r.get('delta', 0):<10.3f}",
        )

    finals = [r["loss_last"] for r in results if r["loss_last"] == r["loss_last"]]
    if finals:
        spread = max(finals) - min(finals)
        LOG.info("=" * 60)
        LOG.info("Spread of final losses: %.3f (gate: %.3f)", spread, args.gate_loss_spread)
        if spread < args.gate_loss_spread:
            LOG.info("VERDICT: spread below gate -> KEEP donut4 (4 layers).")
        else:
            best = min(results, key=lambda r: r["loss_last"])
            LOG.info("VERDICT: best variant = %d layers (loss=%.3f, params=%.1fM)",
                     best["layers"], best["loss_last"], best["params"] / 1e6)


if __name__ == "__main__":
    main()
