"""Tests for the :class:`vista_ocr.data.bbox.BBox` value object.

Pure Python; no torch / albumentations / PIL imports. The
``from_normalised_xywh`` test pins the conversion bit-for-bit against
the legacy ``_norm_bbox_to_pixels`` formula so the refactor that
replaces it is provably equivalent.
"""
from __future__ import annotations

import pytest

from vista_ocr.data.bbox import BBox


class TestConstruction:
    def test_post_init_rejects_inverted_x(self):
        with pytest.raises(ValueError, match="inverted"):
            BBox(10, 0, 5, 10)

    def test_post_init_rejects_inverted_y(self):
        with pytest.raises(ValueError, match="inverted"):
            BBox(0, 10, 10, 5)

    def test_post_init_allows_degenerate_zero_width(self):
        bb = BBox(5, 5, 5, 10)
        assert bb.is_degenerate()

    def test_post_init_allows_degenerate_zero_height(self):
        bb = BBox(5, 5, 10, 5)
        assert bb.is_degenerate()

    def test_frozen_cannot_mutate(self):
        bb = BBox(0, 0, 10, 10)
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            bb.x1 = 99  # type: ignore[misc]

    def test_hashable(self):
        # Must be hashable so it can live in sets / dict keys.
        s = {BBox(0, 0, 10, 10), BBox(0, 0, 10, 10), BBox(1, 1, 2, 2)}
        assert len(s) == 2


class TestConstructors:
    def test_from_xyxy(self):
        assert BBox.from_xyxy(1, 2, 3, 4) == BBox(1, 2, 3, 4)

    def test_from_xywh(self):
        assert BBox.from_xywh(10, 20, 5, 7) == BBox(10, 20, 15, 27)

    def test_from_albumentations_rounds_floats(self):
        bb = BBox.from_albumentations(1.4, 2.6, 9.5, 10.4)
        assert bb == BBox(1, 3, 10, 10)

    def test_from_albumentations_clamps_tiny_inversion(self):
        # Albumentations can return x2 < x1 by 1px after rotation +
        # min_visibility filtering -- clamp rather than raise.
        bb = BBox.from_albumentations(10.0, 0.0, 9.6, 10.0)
        assert bb.x2 == bb.x1
        assert bb.is_degenerate()


class TestFromNormalisedXywh:
    """Bit-for-bit equivalence with the legacy _norm_bbox_to_pixels."""

    def _legacy(self, bbox, img_w, img_h):
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None
        x1 = max(0, int(round(x * img_w)))
        y1 = max(0, int(round(y * img_h)))
        x2 = min(img_w, int(round((x + w) * img_w)))
        y2 = min(img_h, int(round((y + h) * img_h)))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    @pytest.mark.parametrize("nx,ny,nw,nh,W,H", [
        (0.1, 0.2, 0.3, 0.4, 1000, 800),
        (0.0, 0.0, 1.0, 1.0, 100, 100),     # full canvas
        (0.5, 0.5, 0.5, 0.5, 200, 200),     # bottom-right quadrant
        (0.001, 0.001, 0.001, 0.001, 1000, 1000),  # tiny -> rounding
        (0.99, 0.99, 0.02, 0.02, 100, 100),  # over-edge -> clipped
    ])
    def test_matches_legacy(self, nx, ny, nw, nh, W, H):
        legacy = self._legacy([nx, ny, nw, nh], W, H)
        new = BBox.from_normalised_xywh(nx, ny, nw, nh, img_w=W, img_h=H)
        if legacy is None:
            # Legacy filtered (tiny / over-edge); new returns a possibly
            # degenerate BBox -- caller filters via is_degenerate().
            assert new.is_degenerate()
        else:
            assert new.to_xyxy() == legacy

    def test_clips_to_canvas(self):
        bb = BBox.from_normalised_xywh(0.0, 0.0, 2.0, 2.0, img_w=50, img_h=40)
        assert bb == BBox(0, 0, 50, 40)


class TestExports:
    def test_to_xyxy(self):
        assert BBox(1, 2, 3, 4).to_xyxy() == (1, 2, 3, 4)

    def test_to_xywh(self):
        assert BBox(10, 20, 15, 27).to_xywh() == (10, 20, 5, 7)

    def test_to_albumentations_returns_floats(self):
        out = BBox(1, 2, 3, 4).to_albumentations()
        assert out == (1.0, 2.0, 3.0, 4.0)
        assert all(isinstance(v, float) for v in out)


class TestProperties:
    def test_width_height_area(self):
        bb = BBox(0, 0, 10, 5)
        assert bb.width == 10
        assert bb.height == 5
        assert bb.area == 50

    def test_is_degenerate_default_thresholds(self):
        assert BBox(5, 5, 5, 10).is_degenerate()  # zero width
        assert BBox(5, 5, 10, 5).is_degenerate()  # zero height
        assert not BBox(0, 0, 1, 1).is_degenerate()  # 1x1 is fine

    def test_is_degenerate_custom_min(self):
        bb = BBox(0, 0, 5, 5)
        assert bb.is_degenerate(min_w=10)
        assert bb.is_degenerate(min_h=10)
        assert not bb.is_degenerate(min_w=5, min_h=5)


class TestOperations:
    def test_clip_to_inside(self):
        bb = BBox(10, 10, 50, 50).clip_to(img_w=100, img_h=100)
        assert bb == BBox(10, 10, 50, 50)

    def test_clip_to_overflows(self):
        bb = BBox(-5, -5, 110, 110).clip_to(img_w=100, img_h=100)
        assert bb == BBox(0, 0, 100, 100)

    def test_clip_to_collapses_outside_canvas(self):
        # Box entirely past the right edge collapses to a degenerate
        # box at the canvas edge.
        bb = BBox(150, 0, 200, 100).clip_to(img_w=100, img_h=100)
        assert bb.is_degenerate()
        assert bb.x1 == 100 and bb.x2 == 100

    def test_pad_grows_each_side(self):
        bb = BBox(10, 10, 20, 20).pad(3)
        assert bb == BBox(7, 7, 23, 23)

    def test_pad_negative_shrinks(self):
        bb = BBox(10, 10, 20, 20).pad(-2)
        assert bb == BBox(12, 12, 18, 18)

    def test_scale_uniform(self):
        bb = BBox(10, 10, 20, 20).scale(sx=2.0, sy=2.0)
        assert bb == BBox(20, 20, 40, 40)

    def test_scale_anisotropic(self):
        bb = BBox(10, 10, 20, 20).scale(sx=2.0, sy=0.5)
        assert bb == BBox(20, 5, 40, 10)

    def test_contains_inside(self):
        outer = BBox(0, 0, 100, 100)
        inner = BBox(10, 10, 90, 90)
        assert outer.contains(inner)

    def test_contains_self(self):
        bb = BBox(0, 0, 10, 10)
        assert bb.contains(bb)

    def test_contains_overlap_not_inside(self):
        outer = BBox(0, 0, 50, 50)
        partial = BBox(40, 40, 60, 60)
        assert not outer.contains(partial)

    def test_contains_disjoint(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(20, 20, 30, 30)
        assert not a.contains(b)
