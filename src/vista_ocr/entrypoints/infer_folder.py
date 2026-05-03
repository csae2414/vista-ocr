"""``vista-ocr infer`` -- decode every image in a folder.

GT-free: just runs the model and writes one JSONL line per image. The
file starts with a single ``_meta`` record carrying the ckpt path,
ckpt step, vista-ocr version, and timestamp so multiple inference
outputs are traceable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="vista-ocr infer",
        description="Decode every image in a folder; no ground truth required.",
        add_help=add_help,
    )
    ap.add_argument("--folder", required=True, type=Path,
                    help="Directory holding images (recursive scan).")
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument("--out-jsonl", required=True, type=Path)
    ap.add_argument("--page-h", type=int, default=1050)
    ap.add_argument("--page-w", type=int, default=1400)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def _iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES:
            yield p


def run(args: argparse.Namespace) -> int:
    import json
    import logging
    import time
    from datetime import datetime, timezone
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    import torch
    from PIL import Image

    from vista_ocr.inference.generate import InferenceConfig, ocr_with_layout
    from vista_ocr.logging_config import setup_logging
    from vista_ocr.models.decoder import small_random_decoder
    from vista_ocr.models.encoder import FCNEncoderWidther
    from vista_ocr.models.vista_ocr import VistaOCR
    from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
    from vista_ocr.tokenizer.tokenizer import VistaTokenizer
    from vista_ocr.training.callbacks import load_checkpoint

    LOG = logging.getLogger("infer-folder")
    setup_logging(level="INFO")
    torch.manual_seed(args.seed)

    if not args.folder.is_dir():
        raise SystemExit(f"--folder not found: {args.folder}")

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

    try:
        pkg_v = _pkg_version("vista-ocr")
    except PackageNotFoundError:
        pkg_v = "unknown"

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_failed = 0
    t0 = time.perf_counter()

    with args.out_jsonl.open("w", encoding="utf-8") as fh, torch.no_grad():
        # Header _meta record so multiple infer JSONL files are
        # disambiguatable post-hoc.
        meta = {
            "_meta": {
                "ckpt": str(args.ckpt),
                "ckpt_step": payload.step,
                "folder": str(args.folder),
                "vista_ocr_version": pkg_v,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "page_h": args.page_h, "page_w": args.page_w,
                "max_new_tokens": args.max_new_tokens,
            },
        }
        fh.write(json.dumps(meta) + "\n")
        for i, img_path in enumerate(_iter_images(args.folder)):
            if args.max_images is not None and i >= args.max_images:
                break
            try:
                with Image.open(img_path) as img:
                    img_l = img.convert("L")
                    img_l.load()
                lines_out = ocr_with_layout(model, img_l, tokenizer, inf_cfg)
                hyp = " ".join(ln.text for ln in lines_out)
            except Exception:
                LOG.exception("decode failed at %s", img_path)
                hyp = ""
                n_failed += 1
            fh.write(json.dumps({
                "image": str(img_path.relative_to(args.folder)),
                "hyp": hyp,
            }, ensure_ascii=False) + "\n")
            n_written += 1
            if (i + 1) % 25 == 0:
                LOG.info("decoded %d images (failed=%d)", i + 1, n_failed)

    elapsed = time.perf_counter() - t0
    LOG.info(
        "infer: images=%d failed=%d wall=%.1fs out=%s",
        n_written, n_failed, elapsed, args.out_jsonl,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
