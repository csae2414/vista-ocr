"""End-to-end finetuning + eval driver for SROIE / IAM / MAURDOR-EN.

Loads a stage-3 checkpoint, finetunes on the chosen dataset for a small
number of steps, then evaluates and prints CER / WER / DetEval / AP@IoU
in the format we'll put in benchmark.md.

Each dataset's loader is chosen by ``--dataset``; the underlying training
loop and eval are shared.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterable

import torch

from vista_ocr.data.collate import collate
from vista_ocr.data.iam import IamConfig, iter_iam
from vista_ocr.data.maurdor import MaurdorConfig, iter_maurdor
from vista_ocr.data.preprocess import PreprocessConfig
from vista_ocr.data.sroie import SroieConfig, iter_sroie
from vista_ocr.eval.metrics_detection import (
    ap_at_iou_thresholds,
    area_f1,
    deteval,
)
from vista_ocr.eval.metrics_recognition import recognition_metrics
from vista_ocr.inference.generate import InferenceConfig, ocr_with_layout
from vista_ocr.logging_config import setup_logging
from vista_ocr.models.decoder import small_random_decoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import VistaTokenizer
from vista_ocr.training.callbacks import load_checkpoint
from vista_ocr.training.train_loop import TrainConfig, train

LOG = logging.getLogger("finetune")


def _iter_dataset(name: str, root: Path, split: str):
    if name == "sroie":
        yield from iter_sroie(SroieConfig(root=root, split=split))
    elif name == "iam":
        yield from iter_iam(IamConfig(root=root, split=split))
    elif name == "maurdor":
        yield from iter_maurdor(MaurdorConfig(root=root, split=split))
    else:
        raise ValueError(f"unknown dataset {name}")


def _evaluate(
    model: VistaOCR,
    samples: Iterable,
    tokenizer: VistaTokenizer,
    *,
    max_eval: int = 100,
    page_h: int = 1100,
    page_w: int = 850,
) -> dict:
    model.eval()
    inf_cfg = InferenceConfig(
        max_new_tokens=2048, target_h=page_h, target_w=page_w, pad_multiple=32, device="cuda",
    )
    refs_text: list[str] = []
    hyps_text: list[str] = []
    deteval_scores = []
    area_scores = []
    aps = {0.5: [], 0.6: [], 0.7: [], 0.8: []}
    n = 0
    for sample in samples:
        if n >= max_eval:
            break
        n += 1
        try:
            pred_lines = ocr_with_layout(model, sample.image, tokenizer, inf_cfg)
        except Exception as exc:                  # noqa: BLE001
            LOG.warning("inference failed on sample %d: %s", n, exc)
            continue
        refs_text.append(" ".join(ln.text for ln in sample.lines))
        hyps_text.append(" ".join(ln.text for ln in pred_lines))
        gt_boxes = [tuple(map(float, ln.bbox)) for ln in sample.lines]
        pred_boxes = [tuple(map(float, ln.bbox)) for ln in pred_lines]
        deteval_scores.append(deteval(gt_boxes, pred_boxes))
        area_scores.append(area_f1(gt_boxes, pred_boxes, page_shape=sample.image.size[::-1]))
        for t, v in ap_at_iou_thresholds(gt_boxes, pred_boxes).items():
            aps[t].append(v)

    rec = recognition_metrics(refs_text, hyps_text)
    return {
        "n": n,
        "cer": rec.cer, "wer": rec.wer, "f1_word": rec.f1,
        "deteval_p": sum(s.precision for s in deteval_scores) / max(1, len(deteval_scores)),
        "deteval_r": sum(s.recall for s in deteval_scores) / max(1, len(deteval_scores)),
        "deteval_f1": sum(s.f1 for s in deteval_scores) / max(1, len(deteval_scores)),
        "area_f1": sum(area_scores) / max(1, len(area_scores)),
        "ap": {t: (sum(v) / max(1, len(v))) for t, v in aps.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("sroie", "iam", "maurdor"), required=True)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Pretrained vista-ocr checkpoint to start from")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--page-h", type=int, default=1100)
    ap.add_argument("--page-w", type=int, default=850)
    args = ap.parse_args()

    setup_logging(level="INFO")

    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(args.spm), grid=grid)
    encoder = FCNEncoderWidther(input_channels=1, dropout=0.5, gradient_checkpointing=True)
    decoder = small_random_decoder(
        vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=4, n_heads=16,
        ffn_dim=4096, max_position_embeddings=4096,
    )
    model = VistaOCR(encoder=encoder, decoder=decoder).cuda()

    if args.checkpoint and args.checkpoint.exists():
        LOG.info("Loading pretrained weights from %s", args.checkpoint)
        load_checkpoint(args.checkpoint, model=model, optimizer=None,
                        map_location="cuda", strict=False)

    pre_cfg = PreprocessConfig(target_h=args.page_h, target_w=args.page_w, pad_multiple=32)

    def train_stream():
        while True:
            yield from _iter_dataset(args.dataset, args.root, "train")

    cfg = TrainConfig(
        base_lr=args.lr, warmup_steps=200, total_steps=args.steps,
        micro_batch_size=args.batch_size, grad_accum_steps=1, log_every=50,
        lambda_text=0.5, target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
        device="cuda", autocast_dtype=torch.bfloat16, gradient_checkpointing=True,
        adam_betas=(0.9, 0.98), adam_eps=1e-6, label_smoothing=0.1,
    )

    LOG.info("Finetuning on %s for %d steps", args.dataset, args.steps)
    t0 = time.perf_counter()
    train(model=model, sample_stream=train_stream(), tokenizer=tokenizer, cfg=cfg,
          max_steps=args.steps)
    LOG.info("Finetune done in %.1fs", time.perf_counter() - t0)

    LOG.info("Evaluating on %s test split", args.dataset)
    eval_split = "test" if args.dataset != "iam" else "test"
    metrics = _evaluate(model, _iter_dataset(args.dataset, args.root, eval_split),
                        tokenizer, max_eval=200, page_h=args.page_h, page_w=args.page_w)
    LOG.info("=" * 60)
    LOG.info("RESULTS on %s (%d samples)", args.dataset, metrics["n"])
    LOG.info("CER=%.4f  WER=%.4f  word-F1=%.4f", metrics["cer"], metrics["wer"], metrics["f1_word"])
    LOG.info("DetEval P=%.4f R=%.4f F1=%.4f", metrics["deteval_p"], metrics["deteval_r"],
             metrics["deteval_f1"])
    LOG.info("Area-F1=%.4f", metrics["area_f1"])
    for t, v in metrics["ap"].items():
        LOG.info("AP@IoU=%.1f: %.4f", t, v)


if __name__ == "__main__":
    main()
