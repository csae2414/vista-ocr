"""Ablation harness shared by the operator-facing comparison scripts.

The four scripts in ``scripts/ablate_*.py`` and ``scripts/decoder_ab.py``
share the same skeleton -- build N model variants, train each for the
same number of steps, summarise the loss histories, print a side-by-
side table -- with the variant-construction step being the only thing
that genuinely differs. This module factors out the skeleton.
"""
from vista_ocr.ablation.base import Ablation, AblationVariant, summarise_history

__all__ = ["Ablation", "AblationVariant", "summarise_history"]
