"""Helpers for the per-stage scripts to wire up validation.

Hides the boilerplate of producing ``(Batch, ref_str)`` tuples for the
val factory and producing the matching ``decode_fn`` that the run_loop
calls when ``decode_n > 0``.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from vista_ocr.data.collate import Batch, collate
from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa
from vista_ocr.data.preprocess import PreprocessConfig
from vista_ocr.tokenizer.tokenizer import VistaTokenizer

if TYPE_CHECKING:
    from pathlib import Path


LOG = logging.getLogger(__name__)


def pdfa_val_batches(
    val_shard: Path,
    tokenizer: VistaTokenizer,
    pre_cfg: PreprocessConfig,
) -> Iterator[tuple[Batch, str]]:
    """Yield ``(Batch, ref_text)`` tuples for the val factory.

    ``ref_text`` is the concatenation of the sample's line texts -- the
    ground-truth string we'll compare against the decoded prediction.
    """
    for sample in iter_pdfa(PdfaConfig(shards=[str(val_shard)])):
        ref = " ".join(line.text for line in sample.lines)
        yield collate([sample], tokenizer, pre_cfg), ref


def make_val_loss_fn(spatial_ids, lambda_text: float):
    """Return a ``loss_fn(model, item)`` where ``item`` is ``(Batch, ref)``."""
    from vista_ocr.training.losses import combined_loss

    def fn(model, item):
        batch, _ref = item
        device = next(model.parameters()).device
        logits = model(
            batch.images.to(device), batch.decoder_input_ids.to(device),
        )
        out = combined_loss(
            logits=logits,
            labels=batch.labels.to(device),
            spatial_token_ids=spatial_ids,
            lambda_text=lambda_text,
            pad_id=batch.pad_id,
            prompt_mask=batch.prompt_mask.to(device),
        )
        return out.loss

    return fn


def make_val_decode_fn(
    tokenizer: VistaTokenizer,
    *,
    max_new_tokens: int = 512,
    repetition_penalty: float = 1.05,
):
    """Return a ``decode_fn(model, item)`` returning ``(refs, hyps)``.

    Mild repetition penalty is on by default for diagnostic decoding only
    -- it is NOT applied in the loss path or in benchmark eval. Lets us
    see what the model knows even when stuck in early-training n-gram
    loops.
    """

    def fn(model, item):
        batch, ref = item
        device = next(model.parameters()).device
        # Build a fresh OCR-with-layout prompt; bos already in collate but
        # for generation we need to provide it explicitly.
        prompt_ids = [tokenizer.bos_id, *tokenizer.build_ocr_prompt(with_layout=True)]
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out_ids = model.generate(
            images=batch.images.to(device),
            prompt_ids=prompt,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )[0].tolist()
        out_ids = [i for i in out_ids if i != tokenizer.eos_id]
        pred_lines = tokenizer.parse_original_output(out_ids)
        hyp = " ".join(line.text for line in pred_lines)
        return [ref], [hyp]

    return fn
