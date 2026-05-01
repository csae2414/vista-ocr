"""A/B benchmark: inline iter_pdfa vs multi-worker DataLoader.

Reports wall-clock + per-step time for both paths and prints the loss
trajectory of each so we can verify that:

1. workers=0 is bit-exact identical to the inline path (same data,
   same seed, no parallelism reordering).
2. workers>=1 reaches a similar loss range (samples are interleaved
   across workers, so order differs but the distribution is the same).

Run on the VM with a real PDFA shard.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from vista_ocr.data.dataloader import DataLoaderConfig, make_pdfa_dataloader
from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa
from vista_ocr.data.preprocess import PreprocessConfig
from vista_ocr.logging_config import setup_logging
from vista_ocr.models.decoder import small_random_decoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import VistaTokenizer
from vista_ocr.training.train_loop import TrainConfig, train

LOG = logging.getLogger("bench")


def _build_model(vocab: int, use_mbart: bool = False, attn: str = "eager") -> VistaOCR:
    from vista_ocr.models.decoder import MBartDecoder
    enc = FCNEncoderWidther(input_channels=1, dropout=0.0, gradient_checkpointing=True)
    if use_mbart:
        dec = MBartDecoder.from_pretrained_mbart50(
            vocab_size=vocab, decoder_layers=12, max_position_embeddings=4096,
            load_pretrained_body=False,
            attn_implementation=attn,
        )
    else:
        dec = small_random_decoder(
            vocab_size=vocab, d_model=1024, n_layers=4, n_heads=8, ffn_dim=2048,
            max_position_embeddings=4096,
            attn_implementation=attn,
        )
    return VistaOCR(enc, dec)


def _run(
    *,
    shard: Path | list[Path],
    spm: Path,
    steps: int,
    page_h: int,
    page_w: int,
    num_workers: int,
    seed: int,
    use_mbart: bool = False,
    attn: str = "eager",
    compile_mode: str | None = None,
) -> tuple[float, list[float]]:
    torch.manual_seed(seed)
    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(spm), grid=grid)
    model = _build_model(tokenizer.vocab_size, use_mbart=use_mbart, attn=attn)

    pre_cfg = PreprocessConfig(target_h=page_h, target_w=page_w, pad_multiple=32)
    shards_list = [str(p) for p in (shard if isinstance(shard, list) else [shard])]
    if num_workers > 0:
        loader = make_pdfa_dataloader(
            shards=shards_list,
            tokenizer=tokenizer,
            pre_cfg=pre_cfg,
            dl_cfg=DataLoaderConfig(
                micro_batch_size=1,
                num_workers=num_workers,
                prefetch_factor=4,
            ),
        )
        stream = iter(loader)
    else:
        stream = iter_pdfa(PdfaConfig(shards=shards_list))

    train_cfg = TrainConfig(
        base_lr=3e-4, warmup_steps=20, total_steps=steps,
        micro_batch_size=1, grad_accum_steps=1, log_every=1000,
        lambda_text=1.0, target_h=page_h, target_w=page_w, pad_multiple=32,
        device="cuda", autocast_dtype=torch.bfloat16, gradient_checkpointing=True,
        freeze_decoder=True, adam_betas=(0.9, 0.98), adam_eps=1e-6,
        label_smoothing=0.1, compile_model=compile_mode is not None,
    )
    if compile_mode is not None:
        # Set the global default so train_loop's torch.compile call uses our mode.
        import torch._dynamo as dynamo
        dynamo.config.cache_size_limit = 64

    t0 = time.perf_counter()
    history = train(model, stream, tokenizer, train_cfg, max_steps=steps)
    elapsed = time.perf_counter() - t0
    return elapsed, [h.loss for h in history]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, type=Path, nargs="+",
                    help="One or more PDFA shard tar files")
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--page-h", type=int, default=1100)
    ap.add_argument("--page-w", type=int, default=850)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers-list", nargs="+", type=int, default=[0, 4])
    ap.add_argument("--use-mbart", action="store_true",
                    help="Use 12-layer mBART decoder (compute-heavy; tests CPU<->GPU overlap)")
    ap.add_argument("--attn", default="eager", choices=("eager", "sdpa"))
    ap.add_argument("--compile", dest="compile_mode", default=None,
                    choices=(None, "default", "reduce-overhead", "max-autotune"))
    args = ap.parse_args()

    setup_logging(level="INFO")
    results = []
    for nw in args.workers_list:
        LOG.info("=" * 60)
        LOG.info("Run with num_workers=%d", nw)
        elapsed, losses = _run(
            shard=args.shard, spm=args.spm,
            steps=args.steps, page_h=args.page_h, page_w=args.page_w,
            num_workers=nw, seed=args.seed, use_mbart=args.use_mbart,
            attn=args.attn, compile_mode=args.compile_mode,
        )
        results.append((nw, elapsed, losses))
        LOG.info("workers=%d  elapsed=%.1fs  per-step=%.3fs  first/last loss=%.3f/%.3f",
                 nw, elapsed, elapsed / max(1, len(losses)),
                 losses[0] if losses else float("nan"),
                 losses[-1] if losses else float("nan"))

    LOG.info("=" * 60)
    LOG.info("SUMMARY")
    base_time = results[0][1]
    base_losses = results[0][2]
    for nw, elapsed, losses in results:
        speedup = base_time / elapsed if elapsed > 0 else float("inf")
        if nw == args.workers_list[0]:
            match = "(baseline)"
        else:
            n = min(len(base_losses), len(losses))
            max_diff = max(abs(a - b) for a, b in zip(base_losses[:n], losses[:n])) if n else float("nan")
            mean_diff = sum(abs(a - b) for a, b in zip(base_losses[:n], losses[:n])) / max(1, n)
            match = f"max|diff|={max_diff:.4f}  mean|diff|={mean_diff:.4f}"
        LOG.info("workers=%-3d %.1fs  speedup=%.2fx  %s",
                 nw, elapsed, speedup, match)


if __name__ == "__main__":
    main()
