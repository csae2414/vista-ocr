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

from vista_ocr.data.augment import AugmentConfig  # noqa: E402
from vista_ocr.data.collate import collate  # noqa: E402
from vista_ocr.data.mixture import MixedTaskStream, TaskMix  # noqa: E402
from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa  # noqa: E402
from vista_ocr.data.preprocess import PreprocessConfig  # noqa: E402
from vista_ocr.data.split import assert_not_test_shard  # noqa: E402
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
    pdfa_val_batches,
)

LOG = logging.getLogger("stage3")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shards", nargs="+", required=True, type=Path,
                    help="PDFA train shards (paths to .tar files).")
    ap.add_argument("--idl-shards", nargs="+", type=Path, default=None,
                    help="Phase H: when set, mix IDL into stage 3's "
                         "task-relabelled stream alongside PDFA.")
    ap.add_argument("--idl-weight", type=float, default=0.6,
                    help="Phase H: IDL fraction of the stage-3 mix.")
    ap.add_argument("--synth-handwritten", action="store_true",
                    help="Phase J: mix license-clean handwritten synth into the train stream.")
    ap.add_argument("--synth-weight", type=float, default=0.2,
                    help="Synth fraction when --synth-handwritten is set.")
    ap.add_argument("--synth-text-corpus-en", nargs="+", type=Path, default=None,
                    help="Local UTF-8 corpus path(s) for English synth (NEVER downloads).")
    ap.add_argument("--synth-font-dir", type=Path, default=None,
                    help="Override packaged handwritten font dir.")
    ap.add_argument("--synth-task", default="ocr_layout", choices=("ocr", "ocr_layout"),
                    help="Synth-only task tag for the OCR-vs-layout ablation.")
    ap.add_argument("--synth-language", default="en", choices=("en", "fr"))
    ap.add_argument("--val-shard", required=True, type=Path)
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--init-from", type=Path, default=None,
                    help="stage-2 checkpoint to load model weights from")
    ap.add_argument("--steps", type=int, default=70000)
    ap.add_argument("--page-h", type=int, default=None,
                    help="Page canvas height in px. Overrides --page-preset.")
    ap.add_argument("--page-w", type=int, default=None,
                    help="Page canvas width in px. Overrides --page-preset.")
    ap.add_argument("--page-preset", default="medium",
                    choices=("tiny", "small", "medium", "large", "paper", "auto"),
                    help="Page resolution preset. 'auto' queries CUDA VRAM. "
                         "Default 'medium' = 1100x850 (24 GB 3090).")
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
    ap.add_argument("--select-on", default="val_word_f1",
                    choices=("val_loss", "val_word_f1"),
                    help="DS-fix P3: ckpt_best + early-stop metric.")
    ap.add_argument("--decode-n-best", type=int, default=256,
                    help="DS-fix Phase 2: second-pass eval on ckpt_best "
                         "candidates with this many batches. 0 disables.")
    ap.add_argument("--augment", action="store_true",
                    help="B2: enable train-time bbox-aware augmentation.")
    # A3: stage-3 multitask weights. Paper Section 3.3 reads as equal-
    # weight progressive introduction; the defaults reflect that. An
    # operator can pass non-uniform weights to A/B-test rebalancing.
    ap.add_argument("--w-ocr", type=float, default=0.25)
    ap.add_argument("--w-ocr-layout", type=float, default=0.25)
    ap.add_argument("--w-region-ocr", type=float, default=0.25)
    ap.add_argument("--w-find-it", type=float, default=0.25)
    ap.add_argument("--sdpa", action="store_true",
                    help="C3: monkey-patch MBartAttention to use SDPA. "
                         "Runs the ship-gate first; aborts on failure.")
    ap.add_argument("--grad-accum-steps", type=int, default=1,
                    help="Effective batch = micro_batch * accum. 8 is "
                         "paper-comparable on a 24 GB 3090.")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="Disable encoder gradient checkpointing.")
    ap.add_argument("--compile", action="store_true", dest="compile_model",
                    help="torch.compile the model. Falls back to eager.")
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--early-stop-patience", type=int, default=15,
                    help="Stage 3 (multitask) val is noisy; default "
                         "patience higher than stage 2.")
    ap.add_argument("--early-stop-min-delta", type=float, default=0.01)
    ap.add_argument("--early-stop-warmup", type=int, default=5)
    args = ap.parse_args()

    # Refuse the locked PDFA test shard (see vista_ocr.data.split).
    assert_not_test_shard(args.val_shard)
    for s in args.train_shards:
        assert_not_test_shard(s)

    args.out.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=args.out / "stage3.log")

    from vista_ocr.training.resolution import resolve as resolve_resolution
    if args.page_h is None or args.page_w is None:
        res = resolve_resolution(args.page_preset)
        args.page_h = args.page_h or res.height
        args.page_w = args.page_w or res.width
    LOG.info("Page canvas: %d x %d", args.page_h, args.page_w)

    if args.sdpa:
        from vista_ocr.models.sdpa_patch import enable_with_ship_gate
        enable_with_ship_gate(manifest_dir=args.out)

    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(args.spm), grid=grid)

    grad_ckpt = not args.no_grad_ckpt
    encoder = FCNEncoderWidther(input_channels=1, dropout=0.5, gradient_checkpointing=grad_ckpt)
    decoder = small_random_decoder(
        vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=4, n_heads=16,
        ffn_dim=4096, max_position_embeddings=4096,
    )
    model = VistaOCR(encoder=encoder, decoder=decoder).cuda()

    if args.init_from and args.init_from.exists():
        LOG.info("Initialising from stage-2 checkpoint %s", args.init_from)
        load_checkpoint(args.init_from, model=model, optimizer=None,
                        map_location="cuda", strict=False, restore_rng=False)

    aug_cfg = AugmentConfig(enabled=True) if args.augment else None
    pre_cfg = PreprocessConfig(
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
        augment=aug_cfg,
    )
    if aug_cfg is not None:
        LOG.info("B2: train-time augmentation enabled")

    # Stage-3 task mix. Defaults to paper Section 3.3 equal weights.
    weights = {
        "ocr": args.w_ocr,
        "ocr_layout": args.w_ocr_layout,
        "region_ocr": args.w_region_ocr,
        "find_it": args.w_find_it,
    }
    s = sum(weights.values())
    if s <= 0:
        raise ValueError(f"All task weights are zero: {weights}")
    weights = {k: v / s for k, v in weights.items()}
    LOG.info("Stage-3 task mix: %s", {k: round(v, 3) for k, v in weights.items()})
    mix = TaskMix(weights=weights)

    synth_factory = None
    if args.synth_handwritten:
        from vista_ocr.data.synth.factory import HandwrittenSynthFactory
        if not args.synth_text_corpus_en:
            raise SystemExit("--synth-handwritten set but --synth-text-corpus-en is empty.")
        font_paths = None
        if args.synth_font_dir is not None:
            font_paths = sorted(args.synth_font_dir.glob("*.ttf"))
            if not font_paths:
                raise SystemExit(f"No .ttf files in --synth-font-dir {args.synth_font_dir}")
        synth_factory = HandwrittenSynthFactory(
            text_corpus_paths=[Path(p) for p in args.synth_text_corpus_en],
            language=args.synth_language,
            font_paths=font_paths,
            task=args.synth_task,
            canvas_size=(args.page_h, args.page_w),
        )

    idl_w = args.idl_weight if args.idl_shards else 0.0
    synth_w = args.synth_weight if synth_factory is not None else 0.0
    pdfa_w = 1.0 - idl_w - synth_w
    if pdfa_w <= 0:
        raise SystemExit(
            f"Mix-fraction validation: pdfa_weight={pdfa_w:.3f} (must be > 0)."
        )

    # Multitask relabelling needs the raw Sample, so we stream in-process.
    # Phase H/J: mix PDFA + optional IDL + optional synth at the Sample
    # layer (before MixedTaskStream relabelling) via MixedStream.
    def stage3_stream():
        from vista_ocr.data.idl import IdlConfig, iter_idl
        from vista_ocr.data.mixture_stream import MixedStream, MixedStreamSource

        def _pdfa_samples():
            for shard in args.train_shards:
                yield from iter_pdfa(PdfaConfig(shards=[str(shard)]))

        sources = [MixedStreamSource(name="pdfa", stream=_pdfa_samples(), weight=pdfa_w)]
        if args.idl_shards:
            def _idl_samples():
                for shard in args.idl_shards:
                    yield from iter_idl(IdlConfig(shards=[str(shard)]))
            sources.append(MixedStreamSource(name="idl", stream=_idl_samples(), weight=idl_w))
        if synth_factory is not None:
            sources.append(MixedStreamSource(
                name="synth", stream=synth_factory(0), weight=synth_w,
            ))
        if len(sources) > 1:
            LOG.info(
                "Phase H/J stage-3 mix: pdfa=%.2f idl=%.2f synth=%.2f; "
                "pdfa_shards=%d, idl_shards=%d, synth=%s",
                pdfa_w, idl_w, synth_w,
                len(args.train_shards), len(args.idl_shards or []),
                "ON" if synth_factory else "off",
            )
            base = MixedStream(sources=sources, seed=0)
        else:
            base = _pdfa_samples()
        relabelled = MixedTaskStream(base, mix, seed=0)
        for sample in relabelled:
            yield collate([sample], tokenizer, pre_cfg)

    spatial_ids = tokenizer._spatial_ids

    def val_batches_factory():
        return pdfa_val_batches(args.val_shard, tokenizer, pre_cfg)

    cfg = TrainConfig(
        base_lr=args.lr, warmup_steps=2000, total_steps=args.steps,
        micro_batch_size=1, grad_accum_steps=args.grad_accum_steps, log_every=200,
        lambda_text=args.lambda_text,
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
        device="cuda", autocast_dtype=torch.bfloat16, gradient_checkpointing=grad_ckpt,
        compile_model=args.compile_model,
        freeze_decoder=False,
        adam_betas=(0.9, 0.98), adam_eps=1e-6, label_smoothing=0.1,
        # A4
        min_lr_ratio=args.min_lr_ratio,
        # B1
        encoder_dropout_max=args.encoder_dropout_max,
        dropout_T=args.dropout_T,
        checkpoint=CheckpointConfig(
            out_dir=args.out, save_every=args.ckpt_every, keep_last=3,
            select_on=args.select_on,
        ),
        val=ValConfig(every=args.val_every, max_batches=args.val_batches),
        val_batches_factory=val_batches_factory,
        val_loss_fn=make_val_loss_fn(spatial_ids, lambda_text=args.lambda_text),
        # B3
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
