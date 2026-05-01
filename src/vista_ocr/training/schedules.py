"""Learning-rate and dropout schedules."""
from __future__ import annotations

import math


def linear_warmup_cosine(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr_ratio: float = 0.0,
) -> float:
    """Returns the learning rate at ``step``."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return base_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def exponential_dropout(step: int, *, T: float = 5e4) -> float:
    """DANIEL's exponential dropout schedule: ``1 - exp(-step / T)``."""
    return 1.0 - math.exp(-step / T)
