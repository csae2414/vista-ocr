"""Combined VISTA-OCR loss.

Paper (Sec. 3): the model minimises

.. math::
    \\mathcal{L} = \\lambda \\, \\mathcal{L}_{text} + (1 - \\lambda) \\, \\mathcal{L}_{loc}

where both terms are token-level cross-entropy. ``L_text`` is computed only
over text/special tokens, ``L_loc`` only over spatial tokens. Prompt tokens
are masked out of both terms (PLAN §12.5: paper does not state this, our
documented choice).

This module is deliberately tokenizer-agnostic: callers pass in the set of
spatial token ids and the prompt-mask boolean tensor; the loss does the
rest.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

LOG = logging.getLogger(__name__)


@dataclass
class CombinedLossOutput:
    loss: Tensor              # scalar — weighted total
    loss_text: Tensor         # scalar — unweighted text term
    loss_loc: Tensor          # scalar — unweighted location term
    n_text_tokens: int
    n_loc_tokens: int


def combined_loss(
    logits: Tensor,
    labels: Tensor,
    spatial_token_ids: set[int] | torch.Tensor,
    *,
    lambda_text: float = 0.5,
    pad_id: int = 0,
    prompt_mask: Tensor | None = None,
    label_smoothing: float = 0.0,
) -> CombinedLossOutput:
    """Compute the combined loss.

    :param logits: ``(B, T, V)`` raw decoder logits.
    :param labels: ``(B, T)`` ground-truth ids. Use ``pad_id`` for padding
        positions; they are ignored.
    :param spatial_token_ids: ids of ``<x_*>`` / ``<y_*>`` / ``<xy_*>``
        tokens. May be a Python set or a 1-D ``LongTensor``.
    :param lambda_text: weight of the text term. (Paper-default 0.5;
        PLAN §12.6 lists the sweep range.)
    :param pad_id: id used for padding in ``labels``.
    :param prompt_mask: optional ``(B, T)`` bool tensor; ``True`` positions
        are *prompt* tokens whose loss is ignored.
    :param label_smoothing: passed through to :func:`F.cross_entropy`.
    """
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logits {logits.shape} vs labels {labels.shape}")

    b, t, v = logits.shape
    flat_logits = logits.reshape(b * t, v)
    flat_labels = labels.reshape(b * t)

    valid = flat_labels != pad_id
    if prompt_mask is not None:
        valid &= ~prompt_mask.reshape(b * t)

    if isinstance(spatial_token_ids, set):
        sp_ids_t = torch.tensor(sorted(spatial_token_ids), dtype=torch.long, device=labels.device)
    else:
        sp_ids_t = spatial_token_ids.to(labels.device)

    is_spatial = torch.zeros(v, dtype=torch.bool, device=labels.device)
    is_spatial[sp_ids_t] = True
    label_is_spatial = is_spatial[flat_labels]

    text_mask = valid & ~label_is_spatial
    loc_mask = valid & label_is_spatial

    loss_text = _masked_ce(flat_logits, flat_labels, text_mask, label_smoothing)
    loss_loc = _masked_ce(flat_logits, flat_labels, loc_mask, label_smoothing)

    total = lambda_text * loss_text + (1.0 - lambda_text) * loss_loc

    return CombinedLossOutput(
        loss=total,
        loss_text=loss_text.detach(),
        loss_loc=loss_loc.detach(),
        n_text_tokens=int(text_mask.sum().item()),
        n_loc_tokens=int(loc_mask.sum().item()),
    )


def _masked_ce(
    logits: Tensor, labels: Tensor, mask: Tensor, label_smoothing: float
) -> Tensor:
    if not mask.any():
        # Differentiable zero so backward still works on a batch with no
        # tokens of this class (e.g. an OCR-only batch has no spatial tokens).
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits[mask],
        labels[mask],
        reduction="mean",
        label_smoothing=label_smoothing,
    )


__all__ = ["combined_loss", "CombinedLossOutput"]
