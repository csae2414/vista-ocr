"""Image preprocessing shared by every data loader.

Implements the paper's preprocessing rules (Appendix 0.A.1):

* 200 dpi rasterisation (handled upstream when rendering PDFs).
* Documents larger than ``2480 x 3508`` pixels are resized.
* "Non-straight images are rectified" -- a deskew hook is exposed but is
  off by default since it's expensive.
* Filter non-Latin / empty / flipped documents (loader-level decision;
  this module only exposes :func:`is_latin_text` and :func:`is_blank_image`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

LOG = logging.getLogger(__name__)


def is_latin_text(text: str) -> bool:
    """True if every character in ``text`` lies in Basic Latin or Latin
    Extended (U+0000..U+024F), with whitespace allowed.

    Anything beyond that range -- CJK, Arabic, etc. -- fails the check.
    """
    for ch in text:
        if ch.isspace():
            continue
        if ord(ch) > 0x024F:
            return False
    return True


def is_blank_image(img: Image.Image, threshold: float = 0.99) -> bool:
    """True if the image is essentially uniform (all white or all black)."""
    arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    mean = float(arr.mean())
    return mean > threshold or mean < (1.0 - threshold)


@dataclass
class PreprocessConfig:
    target_h: int = 3508
    target_w: int = 2480
    pad_multiple: int = 32
    rectify: bool = False


def resize_to_canvas(
    img: Image.Image,
    cfg: PreprocessConfig,
) -> tuple[Image.Image, float, tuple[int, int]]:
    """Resize ``img`` so it fits within ``(target_h, target_w)`` while
    preserving aspect ratio."""
    img = img.convert("L")
    w, h = img.size
    scale = min(cfg.target_h / h, cfg.target_w / w, 1.0)
    if scale < 1.0:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    else:
        new_w, new_h = w, h
    return img, scale, (new_h, new_w)


def pad_to_multiple(
    img: Image.Image, multiple: int, fill: int = 255
) -> tuple[Image.Image, tuple[int, int]]:
    """Pad ``img`` so its height and width are multiples of ``multiple``.
    The encoder's strided convolutions need this to land cleanly."""
    w, h = img.size
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    if (new_h, new_w) == (h, w):
        return img, (0, 0)
    canvas = Image.new("L", (new_w, new_h), fill)
    canvas.paste(img, (0, 0))
    return canvas, (new_h - h, new_w - w)


def to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a grayscale PIL image to a ``(1, 1, H, W)`` float tensor in [0, 1]."""
    arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
