"""Bbox-aware training augmentation pipeline.

Conservative ranges and bbox-preserving transforms only. Off by default
so the existing data path is bit-exact equivalent. Apply *after*
``resize_to_canvas`` and *before* ``pad_to_multiple`` so the pipeline
sees the final-resolution image.

Operates on :class:`PIL.Image` + a list of
:class:`vista_ocr.tokenizer.tokenizer.Line` objects. Returns the same
shape (PIL, list[Line]); the bboxes are co-transformed by Albumentations.

Defensive design choices:

- ``border_mode=cv2.BORDER_REPLICATE`` for rotation: avoids black borders
  that look like pseudo-text content.
- ``min_visibility=0.5`` on BboxParams: rotation that pushes a line's
  bbox more than half off-canvas drops the line rather than keeping a
  truncated, mislabelled box.
- No perspective transform. The earlier draft included it; bbox
  preservation under perspective is brittle and the gain on document
  OCR is small.
- JPEG quality 70-95 (not 60-95): heavy compression on grayscale
  produces banding that can flip pixels around bbox edges.
- All probabilities default to 0.5 so on average half the samples are
  unaugmented; that keeps the loss surface anchored.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from vista_ocr.data.bbox import BBox
from vista_ocr.tokenizer.tokenizer import Line

LOG = logging.getLogger(__name__)


@dataclass
class AugmentConfig:
    """Conservative paper-Appendix-aligned augmentation config.

    The paper Appendix 0.A.4 names: background markup, slanted text,
    shadow effects, poor resolution. We model "poor resolution" via
    JPEG and mild blur, "slanted text" via a small rotation, and add
    brightness/contrast jitter (not paper-named) to harden the encoder
    against scanned-document inputs.
    """

    enabled: bool = False
    rotate_deg: float = 2.0
    brightness_limit: float = 0.10
    contrast_limit: float = 0.10
    blur_max_sigma: float = 0.8
    jpeg_quality_min: int = 70
    jpeg_quality_max: int = 95
    p_each: float = 0.5
    bbox_min_visibility: float = 0.5


class Augmenter:
    """Bbox-aware augmenter. Disabled instances are no-ops; checking
    ``cfg.enabled`` is the only allocation cost."""

    def __init__(self, cfg: AugmentConfig | None = None) -> None:
        self.cfg = cfg or AugmentConfig()
        self._pipeline = None
        if self.cfg.enabled:
            self._pipeline = self._build_pipeline()

    def _build_pipeline(self):
        import albumentations as A  # noqa: PLC0415
        import cv2  # noqa: PLC0415

        cfg = self.cfg
        # Albumentations expects an odd kernel size for GaussianBlur;
        # derive from the requested sigma.
        blur_kmax = max(3, int(cfg.blur_max_sigma * 6) | 1)
        return A.Compose(
            [
                A.Rotate(
                    limit=cfg.rotate_deg,
                    p=cfg.p_each,
                    border_mode=cv2.BORDER_REPLICATE,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=cfg.brightness_limit,
                    contrast_limit=cfg.contrast_limit,
                    p=cfg.p_each,
                ),
                A.GaussianBlur(blur_limit=(3, blur_kmax), p=cfg.p_each),
                A.ImageCompression(
                    quality_range=(cfg.jpeg_quality_min, cfg.jpeg_quality_max),
                    p=cfg.p_each,
                ),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["line_idx"],
                min_visibility=cfg.bbox_min_visibility,
            ),
        )

    def __call__(
        self, image: Image.Image, lines: list[Line]
    ) -> tuple[Image.Image, list[Line]]:
        if not self.cfg.enabled or self._pipeline is None:
            return image, lines
        if not lines:
            return image, lines

        # Albumentations works on numpy arrays. Grayscale PIL ('L') -> 2D.
        img_np = np.asarray(image.convert("L"), dtype=np.uint8)
        # pascal_voc format = (x1, y1, x2, y2) in pixel space; BBox
        # exports it directly via to_albumentations. Filter degenerate
        # boxes here (x1 >= x2 or y1 >= y2). Two upstream paths can
        # produce them: SROIE annotations with collapsed quads (fixed
        # in iter_sroie at parse time), and PDFA bboxes that were
        # non-degenerate at the original DPI but rounded to zero width
        # / height after ``resize_to_canvas`` shrinks them. Both
        # crash Albumentations with "x_max <= x_min". The text+bbox
        # for the dropped line is still passed to the loss path via
        # the un-augmented Sample; we only skip it on the augment
        # side. ``line_idx`` is used by the Albumentations pipeline
        # to track surviving boxes through transforms.
        bboxes: list[list[float]] = []
        line_idx: list[int] = []
        for i, ln in enumerate(lines):
            x1, y1, x2, y2 = ln.bbox
            if x2 <= x1 or y2 <= y1:
                continue
            bboxes.append(list(BBox.from_xyxy(x1, y1, x2, y2).to_albumentations()))
            line_idx.append(i)

        if not bboxes:
            # All boxes degenerate after resize -- skip augmentation
            # rather than feeding albumentations an empty list (which
            # is also rejected by some transforms).
            return image, lines

        out = self._pipeline(image=img_np, bboxes=bboxes, line_idx=line_idx)
        new_img = Image.fromarray(out["image"], mode="L")

        # Albumentations 1.4.15 has a rare drift bug where label_fields
        # and bboxes diverge in length under min_visibility filtering --
        # extra line_idx entries appear after some pipelines drop a
        # bbox without dropping its label. Hit by Run D 2026-05-05 at
        # step ~11000; supervisor restart-looped until this loosened
        # to strict=False. The shorter list is the truth (Albumentations
        # preserves order across the pipeline); zip stops there. Warn
        # once per call when misalignment is observed.
        out_bboxes = out["bboxes"]
        out_line_idx = out["line_idx"]
        if len(out_bboxes) != len(out_line_idx):
            LOG.warning(
                "Augmenter: bbox/line_idx length drift "
                "(bboxes=%d, line_idx=%d) -- pairing the shorter prefix",
                len(out_bboxes), len(out_line_idx),
            )
        new_lines: list[Line] = []
        for bbox, idx in zip(out_bboxes, out_line_idx, strict=False):
            bb = BBox.from_albumentations(*bbox)
            if bb.is_degenerate():
                continue
            new_lines.append(Line(text=lines[idx].text, bbox=bb.to_xyxy()))
        return new_img, new_lines


__all__ = ["AugmentConfig", "Augmenter"]
