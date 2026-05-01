"""Stage-3 multitask pretraining.

Stage-3 introduces all four tasks at equal weight (paper Table 6/7 +
Section 3.3): ocr / ocr_layout / region_ocr / find_it. Resumes from a
stage-2 checkpoint.

Per PLAN_VM, find_it query lengths are sampled uniformly in [2..11] words
(paper Table 6 reports 2-5/5-8/8-11 buckets). Region-OCR picks a random
existing line bbox as the region prompt.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from vista_ocr.data.collate import collate
from vista_ocr.data.dataloader import DataLoaderConfig, make_pdfa_dataloader
from vista_ocr.data.mixture import MixedTaskStream, TaskMix
from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa
from vista_ocr.data.preprocess import PreprocessConfig
from vista_ocr.logging_config import setup_logging
from vista_ocr.models.decoder import small_random_decoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import VistaTokenizer
from vista_ocr.training.callbacks import (
    CheckpointConfig,
    ValConfig,
    load_checkpoint,
)
from vista_ocr.training.losses import combined_loss
from vista_ocr.training.train_loop import TrainConfig, train

LOG = logging.getLogger("stage3")


def _multitask_sample_stream(loader, mix: TaskMix, seed: int = 0):
    """Wrap a Batch loader so each underlying Sample (one per Batch when
    micro_bs=1) gets a random task assignment from the mixture."""
    import random
    from vista_ocr.data.mixture import relabel_for_task
    rng = random.Random(seed)
    for batch in loader:
        # The DataLoader emits Batches built from a single Sample. We
        # cannot easily relabel after collate, so for stage-3 we'll feed
        # iter_pdfa raw + relabel + collate per batch.
        yield batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shards", nargs="+", required=True, type=Path)
    ap.add_argument("--val-shard", required=True, type=Path)
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--init-from", type=Path, default=None,
                    help="stage-2 checkpoint to load model weights from")
    ap.add_argument("--steps", type=int, default=70000)
    ap.add_argument("--page-h", type=int, default=1100)
    ap.add_argument("--page-w", type=int, default=850)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lambda-text", type=float, default=0.5)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=args.out / "stage3.log")

    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(args.spm), grid=grid)

    encoder = FCNEncoderWidther(input_channels=1, dropout=0.5, gradient_checkpointing=True)
    decoder = small_random_decoder(
        vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=4, n_heads=16,
        ffn_dim=4096, max_position_embeddings=4096,
    )
    model = VistaOCR(encoder=encoder, decoder=decoder).cuda()

    if args.init_from and args.init_from.exists():
        LOG.info("Initialising from stage-2 checkpoint %s", args.init_from)
        load_checkpoint(args.init_from, model=model, optimizer=None,
                        map_location="cuda", strict=False, restore_rng=False)

    pre_cfg = PreprocessConfig(target_h=args.page_h, target_w=args.page_w, pad_multiple=32)

    # Stage-3 task mix (paper Section 3.3): equal weights on the four tasks.
    mix = TaskMix(weights={
        "ocr": 0.25, "ocr_layout": 0.25, "region_ocr": 0.25, "find_it": 0.25,
    })

    # We can't easily relabel batches built off-process by DataLoader
    # workers, so stage-3 streams Samples in-process. Slower than stage-2
    # but the multitask relabelling needs the raw Sample.
    def stage3_stream():
        for shard in args.train_shards:
            cfg_p = PdfaConfig(shards=[str(shard)])
            base = iter_pdfa(cfg_p)
            relabelled = MixedTaskStream(base, mix, seed=0)
            for sample in relabelled:
                yield collate([sample], tokenizer, pre_cfg)

    spatial_ids = tokenizer._spatial_ids

    def val_batches_factory():
        for s in iter_pdfa(PdfaConfig(shards=[str(args.val_shard)])):
            yield collate([s], tokenizer, pre_cfg)

    def val_loss_fn(model, batch):
        device = next(model.parameters()).device
        logits = model(batch.images.to(device), batch.decoder_input_ids.to(device))
        out = combined_loss(
            logits=logits, labels=batch.labels.to(device),
            spatial_token_ids=spatial_ids, lambda_text=args.lambda_text,
            pad_id=batch.pad_id, prompt_mask=batch.prompt_mask.to(device),
        )
        return out.loss

    cfg = TrainConfig(
        base_lr=args.lr, warmup_steps=2000, total_steps=args.steps,
        micro_batch_size=1, grad_accum_steps=1, log_every=200,
        lambda_text=args.lambda_text,
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
        device="cuda", autocast_dtype=torch.bfloat16, gradient_checkpointing=True,
        freeze_decoder=False,
        adam_betas=(0.9, 0.98), adam_eps=1e-6, label_smoothing=0.1,
        checkpoint=CheckpointConfig(
            out_dir=args.out, save_every=args.ckpt_every, keep_last=3,
        ),
        val=ValConfig(every=args.val_every, max_batches=args.val_batches),
        val_batches_factory=val_batches_factory,
        val_loss_fn=val_loss_fn,
    )

    LOG.info("Stage-3 multitask pretraining: %d steps, lambda=%.2f, lr=%.2e",
             args.steps, args.lambda_text, args.lr)
    t0 = time.perf_counter()
    history = train(model, sample_stream=stage3_stream(), tokenizer=tokenizer,
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
