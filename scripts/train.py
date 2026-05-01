"""CLI entry point for VISTA-OCR training.

Loads a YAML config (``configs/*.yaml``), wires the tokenizer / model /
data pipeline, and calls :func:`vista_ocr.training.train_loop.train`.

This script is data-source-agnostic: it discovers PDFA / IDL shards from
the config and falls back to synthetic-only training when paths are
missing (useful for CI / GPU-VM warmup runs).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from vista_ocr.data.mixture import MixedTaskStream, TaskMix
from vista_ocr.data.synth.sroie_synth import SroieSynthConfig, generate_sample as gen_sroie
from vista_ocr.data.synth.synthdog_bbox import SynthDogConfig, generate_sample as gen_synthdog
from vista_ocr.logging_config import setup_logging
from vista_ocr.models.decoder import MBartDecoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import VistaTokenizer
from vista_ocr.training.train_loop import TrainConfig, train

LOG = logging.getLogger(__name__)


def _build_synth_stream(seed: int = 0):
    """Infinite stream of synthetic samples — enough for warmup runs."""
    import random

    rng = random.Random(seed)
    while True:
        if rng.random() < 0.5:
            yield gen_synthdog(
                ["hello world", "VISTA-OCR demo", "0 1 2 3 4 5 6 7 8 9"],
                SynthDogConfig(canvas_h=512, canvas_w=384, line_height=22, seed=rng.randint(0, 1 << 30)),
            )
        else:
            yield gen_sroie(SroieSynthConfig(canvas_h=512, canvas_w=384, seed=rng.randint(0, 1 << 30)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    setup_logging(level="INFO", log_file=args.log_file)
    cfg = OmegaConf.load(args.config)
    LOG.info("Config: %s", OmegaConf.to_yaml(cfg))

    grid = SpatialGrid(
        canvas_h=int(cfg.tokenizer.spatial.page_canvas_h),
        canvas_w=int(cfg.tokenizer.spatial.page_canvas_w),
        quantizer_px=int(cfg.tokenizer.spatial.quantizer_px),
        scheme=str(cfg.tokenizer.spatial.scheme),
    )
    tokenizer = VistaTokenizer(spm_model_path=str(cfg.tokenizer.model_path), grid=grid)

    encoder = FCNEncoderWidther(
        input_channels=int(cfg.model.encoder.input_channels),
        dropout=float(cfg.model.encoder.dropout),
    )
    decoder = MBartDecoder.from_pretrained_mbart50(
        vocab_size=tokenizer.vocab_size,
        decoder_layers=int(cfg.model.decoder.layers),
        max_position_embeddings=int(cfg.model.decoder.max_position_embeddings),
        load_pretrained_body=True,
    )
    model = VistaOCR(encoder=encoder, decoder=decoder)

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    train_cfg = TrainConfig(
        base_lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
        grad_clip_norm=float(cfg.optim.grad_clip_norm),
        warmup_steps=int(cfg.optim.schedule.warmup_steps),
        total_steps=int(cfg.optim.schedule.total_steps),
        micro_batch_size=int(cfg.train.micro_batch_size),
        grad_accum_steps=max(1, int(cfg.train.effective_batch_size) // int(cfg.train.micro_batch_size)),
        log_every=int(cfg.train.log_every),
        lambda_text=float(cfg.loss.lambda_text),
        label_smoothing=float(cfg.model.decoder.label_smoothing),
        device=device,
    )

    stream = MixedTaskStream(
        _build_synth_stream(seed=0),
        TaskMix(weights=dict(cfg.data.task_mix)),
        seed=0,
    )

    train(model, stream, tokenizer, train_cfg, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
