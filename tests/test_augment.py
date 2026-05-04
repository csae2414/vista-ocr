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


# B2 gap closed: tests that exercise the *Augmenter wrapper*, not
# Albumentations directly. These would catch a wrapper bug that the
# direct-Albumentations test above misses (e.g. dropping bboxes,
# swapping xyxy ordering, or losing the line_idx pairing).

def test_wrapper_identity_when_pipeline_returns_unchanged_bboxes():
    """With rotate=0 / no augments triggered, the wrapper round-trips
    bboxes through Albumentations and back to Line tuples unchanged."""
    aug = Augmenter(AugmentConfig(
        enabled=True, rotate_deg=0.0,
        brightness_limit=0.0, contrast_limit=0.0,
        blur_max_sigma=0.0, jpeg_quality_min=95, jpeg_quality_max=95,
        p_each=0.0,  # nothing fires
    ))
    img = Image.new("L", (200, 100), 200)
    lines_in = [
        Line(text="alpha", bbox=(10, 20, 50, 40)),
        Line(text="beta", bbox=(60, 20, 100, 40)),
        Line(text="gamma", bbox=(110, 20, 150, 40)),
    ]
    _, lines_out = aug(img, lines_in)
    # All three survive, in order, with bboxes intact (the round-trip
    # through Albumentations float pascal_voc + BBox.from_albumentations
    # must not corrupt int xyxy values).
    assert [ln.text for ln in lines_out] == ["alpha", "beta", "gamma"]
    assert [ln.bbox for ln in lines_out] == [ln.bbox for ln in lines_in]


def test_wrapper_preserves_text_idx_pairing_under_rotation():
    """B2: rotation must not silently swap text→bbox pairings. We feed
    three labelled boxes, rotate by 2 deg, and assert the texts come
    back paired with bboxes whose centres are within 5 px of the
    analytic rotated centres of the *original* bboxes."""
    import math

    aug = Augmenter(AugmentConfig(
        enabled=True, rotate_deg=2.0,
        brightness_limit=0.0, contrast_limit=0.0,
        blur_max_sigma=0.0, jpeg_quality_min=95, jpeg_quality_max=95,
        # Force rotation only, no other transforms, no min_visibility drops.
        p_each=1.0, bbox_min_visibility=0.0,
    ))
    img = Image.new("L", (200, 100), 200)
    lines_in = [
        Line(text="one",   bbox=(20, 20, 60, 40)),
        Line(text="two",   bbox=(80, 20, 120, 40)),
        Line(text="three", bbox=(140, 20, 180, 40)),
    ]
    out_img, lines_out = aug(img, lines_in)
    # Image unchanged in size.
    assert out_img.size == (200, 100)
    # Same number of lines, same texts, same order (Albumentations does
    # not reorder when min_visibility=0 and no boxes are dropped).
    assert [ln.text for ln in lines_out] == [ln.text for ln in lines_in]

    # Each output bbox's centre must be within 5 px of an analytic
    # rotation around the image centre (200/2, 100/2) = (100, 50).
    cx0, cy0 = 100, 50
    cos_a = math.cos(math.radians(2.0))
    sin_a = math.sin(math.radians(2.0))
    for src, dst in zip(lines_in, lines_out, strict=True):
        sx = (src.bbox[0] + src.bbox[2]) / 2
        sy = (src.bbox[1] + src.bbox[3]) / 2
        # Albumentations' Rotate with border_mode=replicate uses a
        # standard 2D rotation around the image centre.
        ex = cx0 + (sx - cx0) * cos_a - (sy - cy0) * sin_a
        ey = cy0 + (sx - cx0) * sin_a + (sy - cy0) * cos_a
        dx = (dst.bbox[0] + dst.bbox[2]) / 2
        dy = (dst.bbox[1] + dst.bbox[3]) / 2
        assert abs(dx - ex) < 5, f"{src.text}: cx_diff={dx - ex:.2f}"
        assert abs(dy - ey) < 5, f"{src.text}: cy_diff={dy - ey:.2f}"


def test_wrapper_filters_degenerate_output():
    """If BBox.from_albumentations clamps a tiny inversion to
    degenerate, the wrapper must filter it via is_degenerate() rather
    than emit a zero-area Line. We assert this directly by feeding
    the wrapper a synthetic Albumentations output through monkey-patch."""
    aug = Augmenter(AugmentConfig(
        enabled=True, rotate_deg=2.0,
        brightness_limit=0.0, contrast_limit=0.0,
        blur_max_sigma=0.0, jpeg_quality_min=95, jpeg_quality_max=95,
        p_each=0.0,
    ))

    img = Image.new("L", (200, 100), 200)
    real_pipeline = aug._pipeline

    def fake_pipeline(*, image, bboxes, line_idx):
        # Simulate Albumentations returning two bboxes where one has
        # been clamped to a degenerate single-pixel shape.
        return {
            "image": image,
            "bboxes": [(10.0, 10.0, 50.0, 30.0), (60.0, 10.0, 60.4, 30.0)],
            "line_idx": [0, 1],
        }

    aug._pipeline = fake_pipeline  # type: ignore[assignment]
    try:
        _, lines_out = aug(img, [
            Line(text="keep", bbox=(10, 10, 50, 30)),
            Line(text="drop", bbox=(60, 10, 70, 30)),
        ])
    finally:
        aug._pipeline = real_pipeline

    # Only the non-degenerate line survives.
    assert [ln.text for ln in lines_out] == ["keep"]


@pytest.mark.skipif(not _have_alb(), reason="albumentations / cv2 not installed")
def test_wrapper_filters_degenerate_input_bboxes():
    """Lines arriving at the augmenter with x1==x2 (or y1==y2) must be
    dropped at the input boundary, not passed to Albumentations.

    Two upstream paths produce degenerate input boxes:

    * SROIE annotations with collapsed-quad lines (filtered at parse
      time in iter_sroie since fa81352 -- but the augmenter is the
      defence-in-depth boundary for any future loader bug).
    * PDFA bboxes that were non-degenerate at original DPI but
      rounded to zero width or height after ``resize_to_canvas``
      shrinks the image. Hit Run C at step 0 on PDFA shard 0042.

    Albumentations rejects zero-width / zero-height bboxes mid-batch
    with a clear ValueError. A single bad bbox in a worker takes the
    whole DataLoader down.
    """
    aug = Augmenter(AugmentConfig(
        enabled=True, rotate_deg=2.0,
        brightness_limit=0.0, contrast_limit=0.0,
        blur_max_sigma=0.0, jpeg_quality_min=95, jpeg_quality_max=95,
        p_each=0.0,
    ))

    img = Image.new("L", (200, 100), 200)
    # Three lines: two healthy, one zero-width (post-resize PDFA shape).
    lines = [
        Line(text="keep1", bbox=(10, 10, 50, 30)),
        Line(text="degen", bbox=(60, 10, 60, 30)),   # x1 == x2
        Line(text="keep2", bbox=(70, 10, 110, 30)),
    ]
    # Should not raise.
    out_img, out_lines = aug(img, lines)
    assert out_img.size == img.size
    # Augmentation may reorder / drop boxes via min_visibility, so we
    # don't pin the exact survivors -- the load-bearing assertion is
    # that the call survives the degenerate input.
    assert all(ln.bbox[2] > ln.bbox[0] and ln.bbox[3] > ln.bbox[1]
               for ln in out_lines)


@pytest.mark.skipif(not _have_alb(), reason="albumentations / cv2 not installed")
def test_wrapper_returns_unchanged_when_all_input_bboxes_degenerate():
    """Edge case: every line collapses post-resize. Albumentations
    rejects an empty bbox list for some transforms, so the wrapper
    short-circuits to identity (return input unchanged) rather than
    invoking the pipeline with no boxes."""
    aug = Augmenter(AugmentConfig(
        enabled=True, rotate_deg=2.0, p_each=0.0,
    ))
    img = Image.new("L", (200, 100), 200)
    lines = [
        Line(text="degen-x", bbox=(60, 10, 60, 30)),
        Line(text="degen-y", bbox=(10, 50, 50, 50)),
    ]
    out_img, out_lines = aug(img, lines)
    assert out_img is img
    assert out_lines is lines
