"""Page-resolution presets + VRAM-aware auto-selection.

Stage scripts run with a ``--page-h x --page-w`` pair that drives the
encoder's input canvas. Larger canvases see more pixels per line
(better text fidelity) but eat memory roughly with H*W. On a 24 GB
3090 with bf16 + grad checkpointing + batch=1, ~1100x850 fits with
headroom; a 40 GB+ card can push to ~1400x1050; an 8-12 GB card
needs to drop to ~700x550.

This module is pure Python (no torch dependency at import) so it can
be unit-tested cheaply. The ``auto`` resolver queries
:func:`torch.cuda.get_device_properties` lazily.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageResolution:
    height: int
    width: int

    def __str__(self) -> str:
        return f"{self.height}x{self.width}"


# Presets ordered cheapest -> richest. Numbers are conservative against
# bf16 + grad checkpointing + 12-layer mBart decoder + batch_size=1.
PRESETS: dict[str, PageResolution] = {
    "tiny":   PageResolution(height=700,  width=550),   # ~8-12 GB
    "small":  PageResolution(height=900,  width=700),   # ~12-16 GB
    "medium": PageResolution(height=1100, width=850),   # ~20-24 GB (3090)
    "large":  PageResolution(height=1400, width=1050),  # ~40 GB+ (A100/L40s)
}

# (lower_gb_inclusive, preset) -- pick the highest preset whose
# threshold the box satisfies. 0 GB serves as the conservative floor.
_VRAM_THRESHOLDS: list[tuple[int, str]] = [
    (0, "tiny"),
    (14, "small"),
    (20, "medium"),
    (38, "large"),
]


def preset_for_vram(vram_gb: float) -> str:
    """Return the largest preset name whose VRAM threshold is met."""
    chosen = "tiny"
    for threshold, name in _VRAM_THRESHOLDS:
        if vram_gb >= threshold:
            chosen = name
    return chosen


def detect_vram_gb() -> float | None:
    """Return total CUDA VRAM in GB, or None when no CUDA device is visible."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024 ** 3)


def resolve(preset: str | None) -> PageResolution:
    """Map a CLI ``--page-preset`` value to a :class:`PageResolution`.

    ``preset == 'auto'`` queries CUDA VRAM and falls back to ``tiny``
    when no CUDA device is visible. ``preset is None`` means "use the
    stage-script default (medium)".
    """
    if preset is None:
        return PRESETS["medium"]
    if preset == "auto":
        vram = detect_vram_gb()
        if vram is None:
            LOG.warning(
                "VRAM auto-detect: no CUDA device visible, falling back to tiny.",
            )
            return PRESETS["tiny"]
        name = preset_for_vram(vram)
        LOG.info(
            "VRAM auto-detect: %.1f GB -> preset=%s (%s)",
            vram, name, PRESETS[name],
        )
        return PRESETS[name]
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown page preset '{preset}'. "
            f"Choose from: {sorted(PRESETS)} or 'auto'.",
        )
    return PRESETS[preset]


__all__ = [
    "PRESETS", "PageResolution", "detect_vram_gb",
    "preset_for_vram", "resolve",
]
