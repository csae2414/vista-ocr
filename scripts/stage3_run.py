"""Stage-3 multitask pretraining.

Stage-3 introduces all four tasks at equal weight (paper Section 3.3):
ocr / ocr_layout / region_ocr / find_it. Resumes from a stage-2
checkpoint.

Differs from stage-2:

- Multitask sample relabelling via :class:`MixedTaskStream`.
- B1 dropout schedule still on (continues from stage-2's regime).
- A4 ``min_lr_ratio`` floor on the cosine schedule.
- B3 CER/WER reported on the first ``--decode-n`` val batches.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from vista_ocr.data.collate import collate  # noqa: E402
from vista_ocr.data.mixture import MixedTaskStream, TaskMix  # noqa: E402
from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa  # noqa: E402
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

LOG = logging.getLogger("stage3")


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
    ap.add_argument("--min-lr-ratio", type=float, default=0.05,
                    help="A4: cosine-schedule floor as a fraction of base_lr.")
    ap.add_argument("--encoder-dropout-max", type=float, default=0.5,
                    help="B1: peak DANIEL exponential dropout. None disables.")
    ap.add_argument("--dropout-T", type=float, default=5e4,
                    help="B1: time-constant of the dropout schedule.")
    ap.add_argument("--decode-n", type=int, default=5,
                    help="B3: decode + score CER/WER on first N val batches.")
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

    # Multitask relabelling needs the raw Sample, so we stream in-process.
    def stage3_stream():
        for shard in args.train_shards:
            cfg_p = PdfaConfig(shards=[str(shard)])
            base = iter_pdfa(cfg_p)
            relabelled = MixedTaskStream(base, mix, seed=0)
            for sample in relabelled:
                yield collate([sample], tokenizer, pre_cfg)

    spatial_ids = tokenizer._spatial_ids

    def val_batches_factory():
        return pdfa_val_batches(args.val_shard, tokenizer, pre_cfg)

    cfg = TrainConfig(
        base_lr=args.lr, warmup_steps=2000, total_steps=args.steps,
        micro_batch_size=1, grad_accum_steps=1, log_every=200,
        lambda_text=args.lambda_text,
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
        device="cuda", autocast_dtype=torch.bfloat16, gradient_checkpointing=True,
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
