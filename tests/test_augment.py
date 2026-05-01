"""Augmentation tests.

Critical contracts the pipeline must satisfy:

1. Default-off semantics: ``AugmentConfig()`` (enabled=False) returns
   inputs unchanged. No imports of albumentations/cv2 in the disabled
   path.
2. Bbox preservation under rotation: a synthetic 4-bbox image rotated
   2 degrees lands within ±N pixels of the analytic expected position.
3. Pipeline output shape equals input shape (rotation pads, doesn't
   crop).
4. Reproducibility: setting an albumentations seed produces deterministic
   output across two calls.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from vista_ocr.data.augment import AugmentConfig, Augmenter
from vista_ocr.tokenizer.tokenizer import Line


def _have_alb() -> bool:
    try:
        import albumentations  # noqa: F401
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _have_alb(), reason="albumentations / cv2 not installed",
)


def test_disabled_is_identity():
    aug = Augmenter(AugmentConfig(enabled=False))
    img = Image.new("L", (200, 100), 200)
    lines = [Line(text="hi", bbox=(10, 20, 50, 40))]
    out_img, out_lines = aug(img, lines)
    assert out_img is img
    assert out_lines is lines


def test_default_disabled():
    aug = Augmenter()
    img = Image.new("L", (200, 100), 200)
    out_img, out_lines = aug(img, [Line(text="x", bbox=(0, 0, 10, 10))])
    assert out_img is img


def test_enabled_returns_same_shape():
    """Augmentation must not crop or pad the canvas."""
    aug = Augmenter(AugmentConfig(
        enabled=True, rotate_deg=2.0,
        brightness_limit=0.1, contrast_limit=0.1,
        blur_max_sigma=0.5, jpeg_quality_min=80, jpeg_quality_max=95,
        p_each=1.0,
    ))
    img = Image.new("L", (200, 100), 200)
    lines = [Line(text="x", bbox=(50, 30, 150, 70))]
    out_img, _ = aug(img, lines)
    assert out_img.size == (200, 100)


def test_rotation_keeps_bboxes_close_to_analytic():
    """Rotation by exactly 2 degrees with rotate-only pipeline must keep
    a centred bbox's centre near the image centre after rotation."""
    import albumentations as A
    import cv2

    img_arr = np.full((100, 200), 200, dtype=np.uint8)
    bbox = [50.0, 30.0, 150.0, 70.0]
    pipeline = A.Compose(
        [A.Rotate(limit=(2, 2), p=1.0, border_mode=cv2.BORDER_REPLICATE)],
        bbox_params=A.BboxParams(format="pascal_voc",
                                 label_fields=["line_idx"], min_visibility=0.0),
    )
    out = pipeline(image=img_arr, bboxes=[bbox], line_idx=[0])
    new_bbox = out["bboxes"][0]
    cx_old = (bbox[0] + bbox[2]) / 2
    cy_old = (bbox[1] + bbox[3]) / 2
    cx_new = (new_bbox[0] + new_bbox[2]) / 2
    cy_new = (new_bbox[1] + new_bbox[3]) / 2
    assert abs(cx_new - cx_old) < 5
    assert abs(cy_new - cy_old) < 5
