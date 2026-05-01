"""Helpers for the per-stage scripts to wire up validation.

Hides the boilerplate of producing ``(Batch, ref_str)`` tuples for the
val factory and producing the matching ``decode_fn`` that the run_loop
calls when ``decode_n > 0``.

The decoupling is deliberate: ``val_batches_with_refs`` accepts any
iterable of :class:`vista_ocr.data.types.Sample` so it can be unit-
tested without touching the network or a real PDFA shard.
``pdfa_val_batches`` is a thin wrapper that plugs in
:func:`vista_ocr.data.pdfa.iter_pdfa`.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import torch
from torch import nn

from vista_ocr.data.collate import Batch, collate
from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa
from vista_ocr.data.preprocess import PreprocessConfig
from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.tokenizer import VistaTokenizer
from vista_ocr.training.losses import combined_loss

LOG = logging.getLogger(__name__)


# Public type aliases for the val factory + functions wired into TrainConfig.
ValItem = tuple[Batch, str]
ValBatches = Iterable[ValItem]
ValLossFn = Callable[[nn.Module, ValItem], torch.Tensor]
ValDecodeFn = Callable[[nn.Module, ValItem], tuple[list[str], list[str]]]


def val_batches_with_refs(
    samples: Iterable[Sample],
    tokenizer: VistaTokenizer,
    pre_cfg: PreprocessConfig,
) -> Iterator[ValItem]:
    """Yield ``(Batch, ref_text)`` tuples from a stream of samples.

    ``ref_text`` is the whitespace-joined concatenation of the sample's
    line texts -- the ground-truth string we'll compare against the
    decoded prediction during val.
    """
    for sample in samples:
        ref = " ".join(line.text for line in sample.lines)
        yield collate([sample], tokenizer, pre_cfg), ref


def pdfa_val_batches(
    val_shard: Path,
    tokenizer: VistaTokenizer,
    pre_cfg: PreprocessConfig,
) -> Iterator[ValItem]:
    """``val_batches_with_refs`` plugged into a single PDFA shard."""
    return val_batches_with_refs(
        iter_pdfa(PdfaConfig(shards=[str(val_shard)])),
        tokenizer,
        pre_cfg,
    )


def make_val_loss_fn(spatial_ids, lambda_text: float) -> ValLossFn:
    """Return a ``loss_fn(model, item)`` where ``item`` is ``(Batch, ref)``."""

    def fn(model: nn.Module, item: ValItem) -> torch.Tensor:
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
    no_repeat_ngram_size: int = 0,
    min_new_tokens: int = 0,
) -> ValDecodeFn:
    """Return a ``decode_fn(model, item)`` returning ``(refs, hyps)``.

    The mild ``repetition_penalty`` default (1.05) is on for diagnostic
    decoding only -- it is NOT applied in the loss path or in benchmark
    eval. Lets us see what the model knows even when stuck in early-
    training n-gram loops. Stage scripts may override; finetune_eval.py
    keeps everything default-off so reported numbers stay paper-
    comparable.
    """

    def fn(
        model: nn.Module, item: ValItem
    ) -> tuple[list[str], list[str]]:
        batch, ref = item
        device = next(model.parameters()).device
        prompt_ids = [tokenizer.bos_id, *tokenizer.build_ocr_prompt(with_layout=True)]
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out_ids = model.generate(
            images=batch.images.to(device),
            prompt_ids=prompt,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_new_tokens=min_new_tokens,
        )[0].tolist()
        out_ids = [i for i in out_ids if i != tokenizer.eos_id]
        pred_lines = tokenizer.parse_original_output(out_ids)
        hyp = " ".join(line.text for line in pred_lines)
        return [ref], [hyp]

    return fn
