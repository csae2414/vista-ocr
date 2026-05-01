"""Diagnostic for the val-time CUBLAS_STATUS_EXECUTION_FAILED crash.

Reproduces the failure deterministically by:
1. building the same model + tokenizer as stage1
2. iterating PDFA val samples one at a time
3. printing image + token shapes BEFORE each forward
4. running both train (with grad) and eval (no_grad) forwards on each
5. catching the exact failing batch shape

Setting CUDA_LAUNCH_BLOCKING=1 makes the stack point at the right kernel
launch instead of an asynchronous one further down the queue.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from vista_ocr.data.collate import collate
from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa
from vista_ocr.data.preprocess import PreprocessConfig
from vista_ocr.logging_config import setup_logging
from vista_ocr.models.decoder import small_random_decoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import VistaTokenizer

LOG = logging.getLogger("diag")


def main() -> None:
    setup_logging(level="INFO")
    val_shard = Path("data/raw/pdfa/pdfa-eng-train-0003.tar")
    spm = Path("data/processed/vocab/sp_en_16k.model")
    page_h, page_w = 1100, 850

    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(spm), grid=grid)
    encoder = FCNEncoderWidther(input_channels=1, dropout=0.0, gradient_checkpointing=True)
    decoder = small_random_decoder(
        vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=4, n_heads=16,
        ffn_dim=4096, max_position_embeddings=4096,
    )
    model = VistaOCR(encoder=encoder, decoder=decoder).cuda()
    pre_cfg = PreprocessConfig(target_h=page_h, target_w=page_w, pad_multiple=32)

    n = 0
    for s in iter_pdfa(PdfaConfig(shards=[str(val_shard)])):
        n += 1
        batch = collate([s], tokenizer, pre_cfg)
        img = batch.images.cuda()
        tgt = batch.decoder_input_ids.cuda()
        LOG.info(
            "sample=%d  image=%s  decoder_input=%s  lines=%d  source=%s",
            n, tuple(img.shape), tuple(tgt.shape), len(s.lines), s.source,
        )
        # eval-mode forward, mirrors run_validation
        try:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _ = model(img, tgt)
            LOG.info("  -> eval forward ok, peak %.2f GB",
                     torch.cuda.max_memory_allocated() / 1e9)
        except Exception as exc:
            LOG.error("  >>> eval forward CRASHED: %s", exc)
            LOG.error("  shapes: image=%s decoder_input=%s",
                      tuple(img.shape), tuple(tgt.shape))
            raise
        torch.cuda.empty_cache()
        if n >= 20:
            break

    LOG.info("Iterated %d val samples without crash", n)


if __name__ == "__main__":
    main()
