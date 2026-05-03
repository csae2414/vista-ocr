"""``vista-ocr eval --manifest`` -- generic evaluation against a JSONL manifest.

Decodes every doc in the manifest with greedy + the inference helpers
already used by ``scripts/benchmarks/sroie/eval.py`` and
``scripts/eval_pdfa_holdout.py``, computes CER / WER /
SROIE-style word-set F1, writes a JSON sidecar suitable for pasting
into BENCHMARKS.md. Optionally writes a per-doc predictions JSONL for
forensic analysis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="vista-ocr eval",
        description="Evaluate a checkpoint against a JSONL manifest.",
        add_help=add_help,
    )
    ap.add_argument("--manifest", required=True, type=Path,
                    help="JSONL manifest (see vista_ocr.data.manifest).")
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-predictions", type=Path, default=None,
                    help="Per-doc JSONL (image, ref, hyp, cer, wer). "
                         "Default: not written.")
    ap.add_argument("--page-h", type=int, default=1050)
    ap.add_argument("--page-w", type=int, default=1400)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    ap.add_argument("--max-docs", type=int, default=None,
                    help="Cap docs scored. Default: all in manifest.")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def run(args: argparse.Namespace) -> int:
    import json
    import logging
    import time

    import torch

    from vista_ocr.data.manifest import iter_manifest
    from vista_ocr.eval.metrics_recognition import word_exact_prf
    from vista_ocr.inference.generate import InferenceConfig, ocr_with_layout
    from vista_ocr.logging_config import setup_logging
    from vista_ocr.models.decoder import small_random_decoder
    from vista_ocr.models.encoder import FCNEncoderWidther
    from vista_ocr.models.vista_ocr import VistaOCR
    from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
    from vista_ocr.tokenizer.tokenizer import VistaTokenizer
    from vista_ocr.training.callbacks import load_checkpoint

    LOG = logging.getLogger("eval-manifest")
    setup_logging(level="INFO")
    torch.manual_seed(args.seed)

    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(args.spm), grid=grid)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = FCNEncoderWidther(
        input_channels=1, dropout=0.0, gradient_checkpointing=False,
    )
    decoder = small_random_decoder(
        vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=4, n_heads=16,
        ffn_dim=4096, max_position_embeddings=4096,
    )
    model = VistaOCR(encoder=encoder, decoder=decoder).to(device).eval()
    payload = load_checkpoint(
        args.ckpt, model=model, optimizer=None,
        map_location=device, strict=False, restore_rng=False,
    )

    inf_cfg = InferenceConfig(
        max_new_tokens=args.max_new_tokens,
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
        device=device,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    refs: list[str] = []
    hyps: list[str] = []
    images: list[str] = []
    pred_lines: list[dict] = []
    n_empty = 0
    t0 = time.perf_counter()

    with torch.no_grad():
        for i, sample in enumerate(iter_manifest(args.manifest)):
            if args.max_docs is not None and i >= args.max_docs:
                break
            ref = " ".join(line.text for line in sample.lines)
            try:
                lines_out = ocr_with_layout(model, sample.image, tokenizer, inf_cfg)
                hyp = " ".join(ln.text for ln in lines_out)
            except Exception:
                LOG.exception("decode failed at doc %d", i)
                hyp = ""
            refs.append(ref)
            hyps.append(hyp)
            images.append(sample.source)
            if not hyp.strip():
                n_empty += 1
            if args.out_predictions is not None:
                # Per-doc CER for forensic readers; jiwer is fine on
                # a single (ref, hyp) pair.
                from jiwer import cer as _cer, wer as _wer
                pred_lines.append({
                    "image": sample.source, "ref": ref, "hyp": hyp,
                    "cer": float(_cer(ref or " ", hyp or " ")),
                    "wer": float(_wer(ref or " ", hyp or " ")),
                })
            if (i + 1) % 25 == 0:
                LOG.info("decoded %d docs (empty=%d)", i + 1, n_empty)

    elapsed = time.perf_counter() - t0
    p, r, f = word_exact_prf(refs, hyps)

    from jiwer import cer as _cer, wer as _wer
    cer_overall = float(_cer(refs, hyps)) if refs else float("nan")
    wer_overall = float(_wer(refs, hyps)) if refs else float("nan")

    LOG.info(
        "eval: docs=%d empty=%d cer=%.4f wer=%.4f word_f1=%.4f wall=%.1fs",
        len(refs), n_empty, cer_overall, wer_overall, f, elapsed,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "ckpt": str(args.ckpt),
        "ckpt_step": payload.step,
        "manifest": str(args.manifest),
        "n_docs": len(refs),
        "n_empty": n_empty,
        "cer": cer_overall,
        "wer": wer_overall,
        "precision": p,
        "recall": r,
        "word_f1": f,
        "elapsed_s": elapsed,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }, indent=2))
    LOG.info("Wrote %s", args.out_json)

    if args.out_predictions is not None and pred_lines:
        args.out_predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.out_predictions.open("w", encoding="utf-8") as fh:
            for rec in pred_lines:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        LOG.info("Wrote per-doc predictions to %s", args.out_predictions)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
