"""Axis-aligned bounding box value object.

Canonical-format rule
---------------------

1. **Pixel xyxy is the in-memory canonical form** between data
   ingestion (PDFA loader, synthetic generators, etc.) and the
   tokenizer / collate boundary.
2. **Normalised xywh is the on-disk wire format** (PDFA JSON pages
   store ``[x_norm, y_norm, w_norm, h_norm]``).
3. **Plain ``tuple[int, int, int, int]``** is the public type at
   :class:`vista_ocr.tokenizer.tokenizer.Line` and at the tokenizer /
   collate / decoder layer. ``BBox`` does NOT cross that boundary --
   the tokenizer is downstream of all geometry, has solid coverage,
   and changing its API would expand any refactor's blast radius
   beyond review-friendly.

Construct a ``BBox`` at the I/O boundary (loader, augment wrapper),
do geometry on it, export back to a tuple before passing to ``Line``.

Invalid-bbox policy
-------------------

* Inverted boxes (``x2 < x1`` or ``y2 < y1``) are bugs in the caller;
  ``__post_init__`` raises :class:`ValueError`.
* Degenerate boxes (``x2 == x1`` and/or ``y2 == y1``) are *valid* and
  callers filter them via :meth:`is_degenerate`. This matches the
  pre-refactor behaviour of the PDFA loader, which silently dropped
  zero-area lines.
* :meth:`clip_to` clamps to image bounds; it never inverts.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in pixel space, xyxy convention.

    Frozen so it can be hashed and copied cheaply. All operations
    return a new ``BBox``; nothing mutates in place.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(
                f"BBox is inverted: x1={self.x1} y1={self.y1} "
                f"x2={self.x2} y2={self.y2} (x2 must be >= x1, y2 >= y1)",
            )

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> BBox:
        return cls(int(x1), int(y1), int(x2), int(y2))

    @classmethod
    def from_xywh(cls, x: int, y: int, w: int, h: int) -> BBox:
        """Pixel-space (x, y, w, h)."""
        return cls(int(x), int(y), int(x) + int(w), int(y) + int(h))

    @classmethod
    def from_normalised_xywh(
        cls,
        nx: float, ny: float, nw: float, nh: float,
        *,
        img_w: int, img_h: int,
    ) -> BBox:
        """Normalised (x, y, w, h) in [0, 1] -> pixel xyxy.

        Uses banker's rounding via :func:`round` then clamps to image
        bounds so the result is always a valid (possibly degenerate)
        box. Match-bit-for-bit replacement for the legacy
        ``_norm_bbox_to_pixels`` in :mod:`vista_ocr.data.pdfa`.
        """
        x1 = max(0, int(round(nx * img_w)))
        y1 = max(0, int(round(ny * img_h)))
        x2 = min(img_w, int(round((nx + nw) * img_w)))
        y2 = min(img_h, int(round((ny + nh) * img_h)))
        # Clamp inversion (can happen when nw/nh are tiny negatives
        # from upstream noise) so __post_init__ doesn't raise on
        # otherwise-recoverable wire data.
        x2 = max(x2, x1)
        y2 = max(y2, y1)
        return cls(x1, y1, x2, y2)

    @classmethod
    def from_albumentations(
        cls, x1: float, y1: float, x2: float, y2: float,
    ) -> BBox:
        """Round Albumentations float pascal_voc output to int xyxy."""
        ix1 = int(round(x1))
        iy1 = int(round(y1))
        ix2 = int(round(x2))
        iy2 = int(round(y2))
        # Albumentations may return tiny inversions after rotation +
        # min_visibility filtering. Clamp rather than raise so the
        # caller's "skip if degenerate" filter can run.
        ix2 = max(ix2, ix1)
        iy2 = max(iy2, iy1)
        return cls(ix1, iy1, ix2, iy2)

    # ------------------------------------------------------------------
    # exports
    # ------------------------------------------------------------------

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_xywh(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.width, self.height)

    def to_albumentations(self) -> tuple[float, float, float, float]:
        """``pascal_voc`` = (x_min, y_min, x_max, y_max) in pixel space."""
        return (float(self.x1), float(self.y1),
                float(self.x2), float(self.y2))

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def is_degenerate(self, *, min_w: int = 1, min_h: int = 1) -> bool:
        """True when the box is too small to carry signal.

        Default thresholds (``min_w=min_h=1``) match the legacy
        ``x2 <= x1 or y2 <= y1`` filter at every existing call site.
        """
        return self.width < min_w or self.height < min_h

    # ------------------------------------------------------------------
    # operations (pure, return new BBox)
    # ------------------------------------------------------------------

    def clip_to(self, *, img_w: int, img_h: int) -> BBox:
        """Clamp to ``[0, img_w] x [0, img_h]``."""
        return BBox(
            x1=max(0, min(self.x1, img_w)),
            y1=max(0, min(self.y1, img_h)),
            x2=max(0, min(self.x2, img_w)),
            y2=max(0, min(self.y2, img_h)),
        )

    def pad(self, px: int) -> BBox:
        """Grow each side by ``px`` pixels. Negative values shrink."""
        return BBox(self.x1 - px, self.y1 - px, self.x2 + px, self.y2 + px)

    def scale(self, *, sx: float, sy: float) -> BBox:
        """Multiply each coordinate by the matching axis factor."""
        return BBox(
            int(round(self.x1 * sx)), int(round(self.y1 * sy)),
            int(round(self.x2 * sx)), int(round(self.y2 * sy)),
        )

    def contains(self, other: BBox) -> bool:
        """True iff ``other`` is fully inside ``self`` (inclusive)."""
        return (
            self.x1 <= other.x1 and self.y1 <= other.y1
            and self.x2 >= other.x2 and self.y2 >= other.y2
        )


__all__ = ["BBox"]
