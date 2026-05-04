"""SROIE finetune entry point.

Loads a pretrained checkpoint (typically ``checkpoints/stage3/ckpt_best.pt``
from the pretraining chain), trains for ``--steps`` on the SROIE
training split, validates against the SROIE *test* split, writes
``ckpt_best.pt`` selected on val_word_f1 (DS-fix Phase 3).

Usage::

    python scripts/benchmarks/sroie/run.py \\
        --init-from checkpoints/stage3/ckpt_best.pt \\
        --data-root /tmp/SROIE2019 \\
        --spm        data/processed/vocab/sp_en_16k.model \\
        --out        checkpoints/finetune-sroie \\
        --steps 5000 --val-every 250 --ckpt-every 250

Or invoke the chain wrapper to also fire the post-finetune eval::

    ./scripts/benchmarks/sroie/chain.sh
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from vista_ocr.data.augment import AugmentConfig  # noqa: E402
from vista_ocr.data.collate import Batch, collate  # noqa: E402
from vista_ocr.data.preprocess import PreprocessConfig  # noqa: E402
from vista_ocr.data.sroie import SroieConfig, iter_sroie  # noqa: E402
from vista_ocr.logging_config import setup_logging  # noqa: E402
from vista_ocr.models.decoder import small_random_decoder  # noqa: E402
from vista_ocr.models.encoder import FCNEncoderWidther  # noqa: E402
from vista_ocr.models.vista_ocr import VistaOCR  # noqa: E402
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid  # noqa: E402
from vista_ocr.tokenizer.tokenizer import VistaTokenizer  # noqa: E402
from vista_ocr.training.callbacks import (  # noqa: E402
    CheckpointConfig,
    EarlyStopConfig,
    ValConfig,
    load_checkpoint,
)
from vista_ocr.training.train_loop import TrainConfig, train  # noqa: E402
from vista_ocr.training.val_helpers import (  # noqa: E402
    make_val_decode_fn,
    make_val_loss_fn,
    sroie_val_batches,
)

LOG = logging.getLogger("finetune-sroie")


def _iter_sroie_train(root: Path, tokenizer: VistaTokenizer,
                      pre_cfg: PreprocessConfig):
    """Loop SROIE train indefinitely so total_steps is the only stop
    signal -- the train split has 626 samples which is far smaller
    than typical step budgets."""
    while True:
        for sample in iter_sroie(SroieConfig(root=root, split="train")):
            yield collate([sample], tokenizer, pre_cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True, type=Path,
                    help="Pretrained checkpoint to finetune from. "
                         "Typically checkpoints/stage3/ckpt_best.pt.")
    ap.add_argument("--data-root", required=True, type=Path,
                    help="SROIE flat layout root (created by "
                         "scripts/datasets/setup_sroie.sh, or the "
                         "Kaggle 'SROIE datasetv2' src dir).")
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--page-h", type=int, default=None)
    ap.add_argument("--page-w", type=int, default=None)
    ap.add_argument("--page-preset", default="medium",
                    choices=("tiny", "small", "medium", "large", "paper", "auto"))
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-batches", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--keep-last", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="Finetune LR. Paper-style finetune uses a low "
                         "rate (1e-5 to 5e-5) to avoid disturbing the "
                         "pretrained weights.")
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--decode-n", type=int, default=5,
                    help="Per-val cheap decode count.")
    ap.add_argument("--decode-n-best", type=int, default=200,
                    help="Second-pass eval on ckpt_best candidates. "
                         "SROIE test has 347 docs; 200 covers most.")
    ap.add_argument("--select-on", default="val_word_f1",
                    choices=("val_loss", "val_word_f1"))
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--sdpa", action="store_true")
    ap.add_argument("--grad-accum-steps", type=int, default=4,
                    help="Effective batch = micro_batch * accum.")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--compile", action="store_true", dest="compile_model")
    ap.add_argument("--lambda-text", type=float, default=0.5)
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--early-stop-patience", type=int, default=8)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.005)
    ap.add_argument("--early-stop-warmup", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.init_from.exists():
        raise SystemExit(f"--init-from not found: {args.init_from}")
    if not (args.data_root / "train").exists():
        raise SystemExit(
            f"--data-root must contain train/ and test/ subdirs; "
            f"got {args.data_root}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=args.out / "finetune-sroie.log")

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
    encoder = FCNEncoderWidther(
        input_channels=1, dropout=0.5, gradient_checkpointing=grad_ckpt,
    )
    decoder = small_random_decoder(
        vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=4, n_heads=16,
        ffn_dim=4096, max_position_embeddings=4096,
    )
    model = VistaOCR(encoder=encoder, decoder=decoder).cuda()
    LOG.info("Loading pretrained weights from %s", args.init_from)
    load_checkpoint(args.init_from, model=model, optimizer=None,
                    map_location="cuda", strict=False, restore_rng=False)
    LOG.info("Total params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    aug_cfg = AugmentConfig(enabled=True) if args.augment else None
    pre_cfg = PreprocessConfig(
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
        augment=aug_cfg,
    )
    if aug_cfg is not None:
        LOG.info("B2: train-time augmentation enabled")

    spatial_ids = tokenizer._spatial_ids

    def val_batches_factory():
        # SROIE test split is the held-out signal for ckpt_best selection.
        return sroie_val_batches(
            args.data_root, tokenizer, pre_cfg, split="test",
        )

    train_cfg = TrainConfig(
        base_lr=args.lr, warmup_steps=50, total_steps=args.steps,
        micro_batch_size=1, grad_accum_steps=args.grad_accum_steps,
        log_every=50, lambda_text=args.lambda_text,
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
        device="cuda", autocast_dtype=torch.bfloat16,
        gradient_checkpointing=grad_ckpt,
        compile_model=args.compile_model,
        adam_betas=(0.9, 0.98), adam_eps=1e-6, label_smoothing=0.1,
        min_lr_ratio=args.min_lr_ratio,
        checkpoint=CheckpointConfig(
            out_dir=args.out, save_every=args.ckpt_every,
            keep_last=args.keep_last, select_on=args.select_on,
        ),
        val=ValConfig(every=args.val_every, max_batches=args.val_batches),
        val_batches_factory=val_batches_factory,
        val_loss_fn=make_val_loss_fn(spatial_ids, lambda_text=args.lambda_text),
        val_decode_fn=make_val_decode_fn(tokenizer) if args.decode_n > 0 else None,
        val_decode_n=args.decode_n,
        val_decode_n_best=args.decode_n_best,
        early_stop=(
            EarlyStopConfig(
                enabled=True,
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
                warmup_vals=args.early_stop_warmup,
                metric=args.select_on,
            )
            if args.early_stop else None
        ),
    )

    t0 = time.perf_counter()
    history = train(
        model=model,
        sample_stream=_iter_sroie_train(args.data_root, tokenizer, pre_cfg),
        tokenizer=tokenizer, cfg=train_cfg, max_steps=args.steps,
    )
    elapsed = time.perf_counter() - t0
    if history:
        first, last = history[0].loss, history[-1].loss
        LOG.info(
            "DONE: %d steps in %.1fs (%.3fs/step). loss first=%.3f last=%.3f delta=%.3f",
            len(history), elapsed, elapsed / max(1, len(history)),
            first, last, first - last,
        )


if __name__ == "__main__":
    main()
