"""Long-form eval of a trained checkpoint on the held-out PDFA shard.

Loads a checkpoint, runs greedy decoding over ``--max-batches`` of the
held-out validation shard, and prints CER / WER / word-F1 plus a
fraction of empty hypotheses (early-stage decoder collapse signal).

Designed to fill the "PDFA held-out" row of ``BENCHMARKS.md`` after a
stage-3 pretrain finishes -- no licence-restricted data needed.

Examples::

    python scripts/eval_pdfa_holdout.py \\
        --ckpt checkpoints/stage3/ckpt_best.pt \\
        --val-shard data/raw/pdfa/pdfa-eng-train-0119.tar \\
        --spm data/processed/vocab/sp_en_16k.model \\
        --max-batches 100
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from vista_ocr.data.preprocess import PreprocessConfig
from vista_ocr.eval.metrics_recognition import recognition_metrics
from vista_ocr.logging_config import setup_logging
from vista_ocr.models.decoder import small_random_decoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import VistaTokenizer
from vista_ocr.training.callbacks import load_checkpoint
from vista_ocr.training.val_helpers import make_val_decode_fn, pdfa_val_batches

LOG = logging.getLogger("eval_pdfa")


def _debug_one_batch(model, item, tokenizer, min_new_tokens,
                     repetition_penalty, no_repeat_ngram_size, idx):
    """Print prompt, raw out_ids, post-strip out_ids, parser output, plain
    SPM decode of the post-prompt suffix for one batch. Distinguishes a
    parser bug (model emits text but parser drops it) from a real
    generation collapse (model emits prompt + EOS only or repeats a
    single token forever).
    """
    batch, ref = item
    device = next(model.parameters()).device
    prompt_ids = [tokenizer.bos_id, *tokenizer.build_ocr_prompt(with_layout=True)]
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out_ids = model.generate(
        images=batch.images.to(device),
        prompt_ids=prompt,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
        max_new_tokens=512,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        min_new_tokens=min_new_tokens,
    )[0].tolist()

    suffix = out_ids[len(prompt_ids):]
    suffix_no_eos = [i for i in suffix if i != tokenizer.eos_id]
    pieces = [tokenizer.id_to_piece(i) for i in suffix_no_eos[:60]]
    plain_text = tokenizer.decode_ids(suffix_no_eos)
    parsed_lines = tokenizer.parse_original_output(suffix_no_eos)
    parsed_text = " ".join(line.text for line in parsed_lines)

    print(f"\n--- DEBUG batch {idx} ---")
    print(f"ref            : {ref[:120]!r}")
    print(f"prompt_ids     : {prompt_ids}")
    print(f"len(out_ids)   : {len(out_ids)}  (prompt+gen)")
    print(f"len(suffix)    : {len(suffix)}  (post-prompt)")
    print(f"suffix first 30: {suffix[:30]}")
    print(f"suffix last 5  : {suffix[-5:] if len(suffix) >= 5 else suffix}")
    print(f"contains EOS   : {tokenizer.eos_id in suffix}")
    print(f"first 20 pieces: {pieces[:20]}")
    print(f"plain decode   : {plain_text[:200]!r}")
    print(f"parsed lines   : {len(parsed_lines)} -> {parsed_text[:200]!r}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--val-shard", type=Path, required=True)
    ap.add_argument("--spm", type=Path, required=True)
    ap.add_argument("--page-h", type=int, default=1100)
    ap.add_argument("--page-w", type=int, default=850)
    ap.add_argument("--max-batches", type=int, default=100,
                    help="How many val batches to decode.")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Write the result dict as JSON for downstream "
                         "(e.g. BENCHMARKS.md) ingestion.")
    ap.add_argument("--debug-dump", type=int, default=0,
                    help="For the first N batches, print prompt, raw out_ids, "
                         "post-strip out_ids, parser output, and plain SPM "
                         "decode of the post-prompt suffix. Diagnostic.")
    ap.add_argument("--min-new-tokens", type=int, default=0,
                    help="Force the model to emit at least this many tokens.")
    ap.add_argument("--repetition-penalty", type=float, default=1.05,
                    help="HF repetition_penalty. 1.0=off; >1.0 discounts "
                         "already-emitted tokens. 1.3-1.5 typical for OCR.")
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0,
                    help="Block n-grams from repeating. 0=off; 3 forbids any "
                         "3-gram from re-occurring (kills <x_38> attractor).")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    setup_logging(level="INFO")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(args.spm), grid=grid)
    encoder = FCNEncoderWidther(input_channels=1, dropout=0.0,
                                 gradient_checkpointing=False)
    decoder = small_random_decoder(
        vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=4, n_heads=16,
        ffn_dim=4096, max_position_embeddings=4096,
    )
    model = VistaOCR(encoder=encoder, decoder=decoder).to(device).eval()

    LOG.info("Loading checkpoint %s", args.ckpt)
    payload = load_checkpoint(
        args.ckpt, model=model, optimizer=None,
        map_location=str(device), strict=True, restore_rng=False,
    )
    LOG.info("Resumed from step %d", payload.step)

    pre_cfg = PreprocessConfig(
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
    )
    decode_fn = make_val_decode_fn(
        tokenizer,
        min_new_tokens=args.min_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    refs: list[str] = []
    hyps: list[str] = []
    t0 = time.perf_counter()
    n_decoded = 0
    with torch.no_grad():
        for i, item in enumerate(pdfa_val_batches(args.val_shard, tokenizer, pre_cfg)):
            if i >= args.max_batches:
                break
            if i < args.debug_dump:
                _debug_one_batch(
                    model, item, tokenizer,
                    args.min_new_tokens, args.repetition_penalty,
                    args.no_repeat_ngram_size, i,
                )
            try:
                rs, hs = decode_fn(model, item)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:  # noqa: BLE001
                LOG.warning("batch %d skipped: %s", i, exc)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            refs.extend(rs)
            hyps.extend(hs)
            n_decoded += 1
            if (i + 1) % 25 == 0:
                LOG.info("  decoded %d/%d batches", i + 1, args.max_batches)
    elapsed = time.perf_counter() - t0
    n_empty = sum(1 for h in hyps if not h.strip())

    if not refs:
        LOG.error("No batches decoded -- shard may be empty.")
        raise SystemExit(1)

    rec = recognition_metrics(refs, hyps)
    result = {
        "ckpt": str(args.ckpt),
        "ckpt_step": int(payload.step),
        "val_shard": str(args.val_shard),
        "n_decoded": n_decoded,
        "n_empty": n_empty,
        "empty_frac": n_empty / max(n_decoded, 1),
        "cer": rec.cer,
        "wer": rec.wer,
        "word_f1": rec.f1,
        "elapsed_s": round(elapsed, 1),
    }
    LOG.info("=" * 60)
    LOG.info("PDFA hold-out eval (max_batches=%d, decoded=%d, empty=%d)",
             args.max_batches, n_decoded, n_empty)
    LOG.info("  CER     : %.4f", rec.cer)
    LOG.info("  WER     : %.4f", rec.wer)
    LOG.info("  word-F1 : %.4f", rec.f1)
    LOG.info("  empty   : %d / %d (%.1f %%)",
             n_empty, n_decoded, 100 * n_empty / max(n_decoded, 1))
    LOG.info("  wall    : %.1fs", elapsed)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        LOG.info("Wrote %s", args.out_json)


if __name__ == "__main__":
    main()
