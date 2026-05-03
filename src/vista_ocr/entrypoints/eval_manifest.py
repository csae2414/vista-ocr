"""``vista-ocr eval --manifest`` -- generic evaluation against a JSONL manifest.

Decodes every doc in the manifest with greedy + the inference helpers
already used by ``scripts/benchmarks/sroie/eval.py`` and
``scripts/eval_pdfa_holdout.py``, then computes the metric blocks
applicable to the manifest's shape:

* **Always**: recognition metrics (CER, WER, word-F1) -- paper Table 2
  rows for SROIE; the long-form benchmark side.
* **When ``bboxes`` is in the manifest**: detection metrics (DetEval
  P/R/F1, Area-F1, AP @ IoU 0.5-0.8) -- paper §4.1 detection rows for
  SROIE / IAM / MAURDOR. The ``--bbox-expand-px`` flag applies the
  paper §4.1.1 +1/+2 px expansion to PREDICTED boxes only (the
  asymmetry is enforced by tests).
* **When ``--cer-ap-thresholds`` is set**: AP at CER thresholds (paper
  §4.2 region-OCR row).

Strict manifest homogeneity: the field set of the first record IS the
contract. Any subsequent record with a different field set is rejected
with a clear error, so an operator who accidentally interleaved two
benchmarks in one manifest sees the mistake immediately rather than
through a misleading mixed-metric output.
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
    # G2: detection-side flags. Active only when the manifest carries
    # ``bboxes``; ignored otherwise.
    ap.add_argument("--bbox-expand-px", type=int, default=0,
                    help="Expand PREDICTED boxes by this many px on each "
                         "side before detection-metric scoring. Paper "
                         "Section 4.1.1 reports SROIE numbers with +1 / "
                         "+2 px expansion. Default 0.")
    ap.add_argument("--detection-iou-threshold", type=float, default=0.5,
                    help="IoU threshold used by AP@IoU's lowest tier and "
                         "by Area-F1's matching. Default 0.5 (SROIE Task 1).")
    # G2: region-OCR-side flag.
    ap.add_argument("--cer-ap-thresholds", type=str, default="",
                    help="Comma-separated list of CER thresholds (e.g. "
                         "'0.0,0.1,0.2,0.3'). When set, the sidecar "
                         "carries an AP-at-CER block (paper Section 4.2 "
                         "region-OCR row). Default: not computed.")
    return ap


def _peek_first_record(path: Path) -> dict:
    """Read the first non-blank, non-comment line of the manifest and
    return its parsed dict. Used to determine which optional fields
    are present so the eval verb can dispatch metric blocks."""
    import json as _json

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            return _json.loads(raw)
    raise ValueError(f"manifest is empty or all-comment: {path}")


def run(args: argparse.Namespace) -> int:
    import json
    import logging
    import time

    import torch

    from vista_ocr.data.manifest import iter_manifest_records
    from vista_ocr.eval.metrics_detection import (
        ap_at_iou_thresholds,
        area_f1,
        deteval,
        expand_box,
    )
    from vista_ocr.eval.metrics_recognition import (
        cer_ap as _cer_ap,
        compute_cer,
        compute_wer,
        per_doc_cer,
        word_exact_prf,
    )
    from vista_ocr.eval.sidecar import (
        CerApBlock,
        DetectionBlock,
        EvalSidecar,
    )
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

    # ------------------------------------------------------------------
    # 1. Field-set discovery: peek at the first record.
    # ------------------------------------------------------------------
    first_rec = _peek_first_record(args.manifest)
    expected_fields = set(first_rec.keys())
    has_bboxes = "bboxes" in expected_fields
    cer_ap_thresholds: tuple[float, ...] | None = None
    if args.cer_ap_thresholds:
        try:
            cer_ap_thresholds = tuple(
                float(t) for t in args.cer_ap_thresholds.split(",") if t.strip()
            )
        except ValueError as e:
            raise SystemExit(
                f"--cer-ap-thresholds must be comma-separated floats; "
                f"got {args.cer_ap_thresholds!r}"
            ) from e

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
    pred_lines: list[dict] = []
    n_empty = 0

    # Per-doc detection state, populated only when has_bboxes.
    deteval_scores: list = []
    area_scores: list[float] = []
    ap_per_doc: dict[float, list[float]] = {0.5: [], 0.6: [], 0.7: [], 0.8: []}

    t0 = time.perf_counter()

    with torch.no_grad():
        for i, (sample, raw_record) in enumerate(iter_manifest_records(args.manifest)):
            if args.max_docs is not None and i >= args.max_docs:
                break

            # 2. Strict homogeneity: any field-set mismatch is an error.
            actual_fields = set(raw_record.keys())
            if actual_fields != expected_fields:
                missing = expected_fields - actual_fields
                extra = actual_fields - expected_fields
                raise SystemExit(
                    f"manifest line {i + 1}: heterogeneous record. "
                    f"Expected fields {sorted(expected_fields)}; "
                    f"missing={sorted(missing)} extra={sorted(extra)}. "
                    "All records in a manifest must share the same field "
                    "set. Split into separate manifests if you need a mix."
                )

            ref = " ".join(line.text for line in sample.lines)
            try:
                lines_out = ocr_with_layout(model, sample.image, tokenizer, inf_cfg)
                hyp = " ".join(ln.text for ln in lines_out)
            except Exception:
                LOG.exception("decode failed at doc %d", i)
                hyp = ""
                lines_out = []
            refs.append(ref)
            hyps.append(hyp)
            if not hyp.strip():
                n_empty += 1
            if args.out_predictions is not None:
                pred_lines.append({
                    "image": sample.source, "ref": ref, "hyp": hyp,
                    "cer": per_doc_cer(ref, hyp),
                    "wer": float(compute_wer([ref or " "], [hyp or " "])),
                })

            if has_bboxes:
                # GT comes from sample.lines (already AABB-projected by
                # iter_manifest's record parser).
                gt_boxes = [tuple(map(float, ln.bbox)) for ln in sample.lines]
                # Predictions: expand by --bbox-expand-px (asymmetry --
                # GT is NEVER expanded; doing so would inflate F1 above
                # what's achievable).
                pred_boxes = [
                    tuple(map(float, expand_box(ln.bbox, args.bbox_expand_px)))
                    for ln in lines_out
                ]
                page_shape = (
                    sample.image.size[1], sample.image.size[0],
                )  # (H, W) for area_f1's rasteriser
                deteval_scores.append(deteval(gt_boxes, pred_boxes))
                area_scores.append(area_f1(gt_boxes, pred_boxes, page_shape=page_shape))
                doc_aps = ap_at_iou_thresholds(gt_boxes, pred_boxes)
                for t, v in doc_aps.items():
                    ap_per_doc.setdefault(t, []).append(v)

            if (i + 1) % 25 == 0:
                LOG.info("decoded %d docs (empty=%d)", i + 1, n_empty)

    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # 3. Aggregate metrics.
    # ------------------------------------------------------------------
    p, r, f = word_exact_prf(refs, hyps)
    cer_overall = compute_cer(refs, hyps) if refs else float("nan")
    wer_overall = compute_wer(refs, hyps) if refs else float("nan")

    detection_block = None
    if has_bboxes and deteval_scores:
        n = len(deteval_scores)
        detection_block = DetectionBlock(
            deteval_precision=sum(s.precision for s in deteval_scores) / n,
            deteval_recall=sum(s.recall for s in deteval_scores) / n,
            deteval_f1=sum(s.f1 for s in deteval_scores) / n,
            area_f1=sum(area_scores) / n,
            ap_at_iou={
                f"{t:.1f}": (sum(vs) / len(vs) if vs else 0.0)
                for t, vs in sorted(ap_per_doc.items())
            },
            iou_threshold=args.detection_iou_threshold,
            bbox_expand_px=args.bbox_expand_px,
        )

    cer_ap_block = None
    if cer_ap_thresholds is not None and refs:
        ap_dict = _cer_ap(refs, hyps, thresholds=cer_ap_thresholds)
        cer_ap_block = CerApBlock(
            ap={f"{t:.2f}": v for t, v in sorted(ap_dict.items())},
            thresholds=list(cer_ap_thresholds),
        )

    sidecar = EvalSidecar(
        ckpt=str(args.ckpt),
        ckpt_step=payload.step,
        manifest=str(args.manifest),
        n_docs=len(refs),
        n_empty=n_empty,
        cer=cer_overall,
        wer=wer_overall,
        precision=p,
        recall=r,
        word_f1=f,
        elapsed_s=elapsed,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        detection=detection_block,
        cer_ap=cer_ap_block,
    )

    LOG.info(
        "eval: docs=%d empty=%d cer=%.4f wer=%.4f word_f1=%.4f wall=%.1fs",
        len(refs), n_empty, cer_overall, wer_overall, f, elapsed,
    )
    if detection_block is not None:
        LOG.info(
            "detection: deteval_f1=%.4f area_f1=%.4f bbox_expand=%dpx",
            detection_block.deteval_f1, detection_block.area_f1,
            detection_block.bbox_expand_px,
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(sidecar.to_jsonable(), indent=2))
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
