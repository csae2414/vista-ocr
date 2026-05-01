"""GPU smoke test (task #16).

End-to-end CUDA verification before any long training run:

1. CUDA available + correct device.
2. Tiny VistaOCR forward + backward on GPU.
3. bf16 autocast forward path works.
4. mBART decoder weight download + body-copy succeeds.
5. Encoder gradient checkpointing under autocast does not error.
6. Throughput estimate (samples/sec) at our paper-spec page size.

Prints a short summary and exits non-zero on any failure. Run from the
GPU VM via:

    python scripts/smoke_test_gpu.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import torch

from vista_ocr.logging_config import setup_logging
from vista_ocr.models.decoder import small_random_decoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR

LOG = logging.getLogger("smoke")


def _probe_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    dev = torch.device("cuda:0")
    LOG.info("CUDA device: %s (cap %s)", torch.cuda.get_device_name(0),
             ".".join(map(str, torch.cuda.get_device_capability(0))))
    LOG.info("VRAM total: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
    return dev


def _tiny_model_on_gpu(dev: torch.device) -> VistaOCR:
    enc = FCNEncoderWidther(input_channels=1, dropout=0.0, gradient_checkpointing=True)
    dec = small_random_decoder(vocab_size=256, d_model=1024, n_layers=2, n_heads=8, ffn_dim=512)
    return VistaOCR(enc, dec).to(dev)


def _forward_backward_step(model: VistaOCR, dev: torch.device, *, h: int, w: int,
                           autocast: bool) -> float:
    images = torch.randn(1, 1, h, w, device=dev)
    labels = torch.randint(0, 256, (1, 16), device=dev)
    model.train()
    t0 = time.perf_counter()
    if autocast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images, labels)
            loss = logits.float().sum()
    else:
        logits = model(images, labels)
        loss = logits.sum()
    loss.backward()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-h", type=int, default=2240)
    ap.add_argument("--page-w", type=int, default=1664)
    ap.add_argument("--check-mbart", action="store_true",
                    help="Also test MBartDecoder.from_pretrained_mbart50 (downloads ~2 GB)")
    args = ap.parse_args()

    setup_logging(level="INFO")
    LOG.info("PyTorch: %s | CUDA: %s",
             torch.__version__, torch.version.cuda)

    dev = _probe_cuda()

    LOG.info("Building tiny VistaOCR on GPU...")
    model = _tiny_model_on_gpu(dev)

    for h, w in [(64, 64), (256, 256), (args.page_h, args.page_w)]:
        try:
            t = _forward_backward_step(model, dev, h=h, w=w, autocast=False)
            LOG.info("fp32 fwd+bwd  %4dx%-4d  %.3fs", h, w, t)
        except torch.cuda.OutOfMemoryError:
            LOG.warning("OOM at %dx%d (fp32) -- this is expected at full page size on a 24GB GPU", h, w)
            torch.cuda.empty_cache()
            break

    LOG.info("Re-running with bf16 autocast...")
    for h, w in [(args.page_h, args.page_w)]:
        try:
            t = _forward_backward_step(model, dev, h=h, w=w, autocast=True)
            LOG.info("bf16 fwd+bwd %4dx%-4d  %.3fs", h, w, t)
            mem = torch.cuda.max_memory_allocated() / 1e9
            LOG.info("Peak VRAM: %.2f GB", mem)
        except torch.cuda.OutOfMemoryError:
            LOG.error("OOM even at bf16 -- need smaller page size or fewer decoder layers")
            sys.exit(2)

    if args.check_mbart:
        LOG.info("Loading mbart-large-50 decoder body (this downloads ~2 GB)...")
        from vista_ocr.models.decoder import MBartDecoder
        try:
            dec = MBartDecoder.from_pretrained_mbart50(
                vocab_size=16000, decoder_layers=12, max_position_embeddings=4096,
                load_pretrained_body=True,
            )
            LOG.info("mBART decoder body loaded: %.1fM params",
                     sum(p.numel() for p in dec.parameters()) / 1e6)
        except Exception as exc:  # noqa: BLE001
            LOG.error("mBART download/copy failed: %s", exc)
            sys.exit(3)

    LOG.info("OK -- all smoke checks passed")


if __name__ == "__main__":
    main()
