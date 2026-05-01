"""CLI entry point for VISTA-OCR evaluation.

Loads a checkpoint, runs :func:`vista_ocr.inference.generate.ocr_with_layout`
over an evaluation set, and prints CER/WER/word-F1 plus DetEval/AP/Area-F1.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

from vista_ocr.eval.metrics_detection import (
    ap_at_iou_thresholds,
    area_f1,
    deteval,
)
from vista_ocr.eval.metrics_recognition import recognition_metrics
from vista_ocr.inference.generate import InferenceConfig, ocr_with_layout
from vista_ocr.logging_config import setup_logging
from vista_ocr.models.decoder import MBartDecoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import Line, VistaTokenizer

LOG = logging.getLogger(__name__)


def _load_eval_set(path: Path) -> list[tuple[Image.Image, list[Line]]]:
    """Eval JSON format: list of {image: <relpath>, lines: [{text, bbox}]}."""
    items = json.loads(path.read_text())
    out = []
    for item in items:
        img = Image.open(path.parent / item["image"]).convert("L")
        lines = [Line(text=ln["text"], bbox=tuple(ln["bbox"])) for ln in item["lines"]]
        out.append((img, lines))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--eval-set", type=Path, required=True)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    setup_logging(level="INFO")
    cfg = OmegaConf.load(args.config)

    grid = SpatialGrid(
        canvas_h=int(cfg.tokenizer.spatial.page_canvas_h),
        canvas_w=int(cfg.tokenizer.spatial.page_canvas_w),
        quantizer_px=int(cfg.tokenizer.spatial.quantizer_px),
        scheme=str(cfg.tokenizer.spatial.scheme),
    )
    tokenizer = VistaTokenizer(str(cfg.tokenizer.model_path), grid)

    encoder = FCNEncoderWidther(input_channels=int(cfg.model.encoder.input_channels))
    decoder = MBartDecoder.from_pretrained_mbart50(
        vocab_size=tokenizer.vocab_size,
        decoder_layers=int(cfg.model.decoder.layers),
        max_position_embeddings=int(cfg.model.decoder.max_position_embeddings),
        load_pretrained_body=False,
    )
    model = VistaOCR(encoder, decoder)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    model.to(device).eval()
    inf_cfg = InferenceConfig(device=device)

    refs_text: list[str] = []
    hyps_text: list[str] = []
    deteval_scores = []
    area_scores = []
    aps = {0.5: [], 0.6: [], 0.7: [], 0.8: []}

    for img, gt_lines in _load_eval_set(args.eval_set):
        pred_lines = ocr_with_layout(model, img, tokenizer, inf_cfg)
        refs_text.append(" ".join(ln.text for ln in gt_lines))
        hyps_text.append(" ".join(ln.text for ln in pred_lines))
        gt_boxes = [tuple(map(float, ln.bbox)) for ln in gt_lines]
        pred_boxes = [tuple(map(float, ln.bbox)) for ln in pred_lines]
        deteval_scores.append(deteval(gt_boxes, pred_boxes))
        area_scores.append(area_f1(gt_boxes, pred_boxes, page_shape=img.size[::-1]))
        for t, v in ap_at_iou_thresholds(gt_boxes, pred_boxes).items():
            aps[t].append(v)

    rec = recognition_metrics(refs_text, hyps_text)
    LOG.info(
        "Recognition: CER=%.4f WER=%.4f F1=%.4f (P=%.4f R=%.4f)",
        rec.cer, rec.wer, rec.f1, rec.precision, rec.recall,
    )
    LOG.info(
        "Detection (DetEval): P=%.4f R=%.4f F1=%.4f",
        sum(s.precision for s in deteval_scores) / max(1, len(deteval_scores)),
        sum(s.recall for s in deteval_scores) / max(1, len(deteval_scores)),
        sum(s.f1 for s in deteval_scores) / max(1, len(deteval_scores)),
    )
    LOG.info("Area-F1: %.4f", sum(area_scores) / max(1, len(area_scores)))
    for t, vs in aps.items():
        LOG.info("AP@IoU=%.1f: %.4f", t, sum(vs) / max(1, len(vs)))


if __name__ == "__main__":
    main()
