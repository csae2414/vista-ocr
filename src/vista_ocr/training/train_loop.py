"""Training loop for VISTA-OCR.

Pure PyTorch (no Accelerate) so the same code runs on CPU during dev and
single-GPU on the VM. The DDP / multi-GPU path is intentionally deferred
until we measure single-GPU throughput on the actual hardware.

The loop:

1. iterates over a list of :class:`vista_ocr.data.types.Sample`s in mini
   batches collated via :func:`vista_ocr.data.collate.collate`;
2. computes the combined loss with prompt-token masking;
3. supports gradient accumulation to reach the paper's effective batch
   size of 1111;
4. applies linear-warmup + cosine LR schedule.

Optimizer is AdamW (paper says "Adam weighted"). Mixed precision uses
bf16 when available, fp32 on CPU.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from vista_ocr.data.collate import Batch, collate
from vista_ocr.data.preprocess import PreprocessConfig
from vista_ocr.data.types import Sample
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.tokenizer import VistaTokenizer
from vista_ocr.training.losses import combined_loss
from vista_ocr.training.schedules import linear_warmup_cosine

LOG = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    base_lr: float = 5e-5
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    warmup_steps: int = 5000
    total_steps: int = 200000
    micro_batch_size: int = 1
    grad_accum_steps: int = 1
    log_every: int = 50
    lambda_text: float = 0.5
    label_smoothing: float = 0.0
    pad_multiple: int = 32
    target_h: int = 3508
    target_w: int = 2480
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    freeze_decoder: bool = False


@dataclass
class StepStats:
    step: int
    loss: float
    loss_text: float
    loss_loc: float
    lr: float
    n_text_tokens: int
    n_loc_tokens: int


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim == 1 or n.endswith(".bias") else decay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def _iter_batches(
    samples: Iterable[Sample],
    tokenizer: VistaTokenizer,
    pre_cfg: PreprocessConfig,
    micro_bs: int,
) -> Iterator[Batch]:
    buf: list[Sample] = []
    for s in samples:
        buf.append(s)
        if len(buf) == micro_bs:
            yield collate(buf, tokenizer, pre_cfg)
            buf = []
    if buf:
        yield collate(buf, tokenizer, pre_cfg)


def train(
    model: VistaOCR,
    sample_stream: Iterable[Sample],
    tokenizer: VistaTokenizer,
    cfg: TrainConfig,
    on_step: Callable[[StepStats], None] | None = None,
    max_steps: int | None = None,
) -> list[StepStats]:
    """Train ``model`` on ``sample_stream``.

    :param sample_stream: iterable yielding :class:`Sample`s. Should be
        large enough (or repeating) to cover ``total_steps`` * micro batches
        * grad accum.
    :param max_steps: if set, stop after this many optimisation steps. Used
        by tests to keep the loop short.
    """
    device = torch.device(cfg.device)
    model.to(device)
    if cfg.freeze_decoder:
        model.freeze_decoder(True)
    optimizer = make_optimizer(model, cfg)
    spatial_ids = tokenizer._spatial_ids
    pre_cfg = PreprocessConfig(
        target_h=cfg.target_h, target_w=cfg.target_w, pad_multiple=cfg.pad_multiple
    )

    history: list[StepStats] = []
    accum = 0
    optimizer.zero_grad(set_to_none=True)
    step = 0

    micro_iter = _iter_batches(
        sample_stream, tokenizer, pre_cfg, cfg.micro_batch_size
    )

    for batch in micro_iter:
        if max_steps is not None and step >= max_steps:
            break
        batch_device = Batch(
            images=batch.images.to(device),
            decoder_input_ids=batch.decoder_input_ids.to(device),
            labels=batch.labels.to(device),
            prompt_mask=batch.prompt_mask.to(device),
            pad_id=batch.pad_id,
        )

        logits = model(batch_device.images, batch_device.decoder_input_ids)
        out = combined_loss(
            logits=logits,
            labels=batch_device.labels,
            spatial_token_ids=spatial_ids,
            lambda_text=cfg.lambda_text,
            pad_id=batch_device.pad_id,
            prompt_mask=batch_device.prompt_mask,
            label_smoothing=cfg.label_smoothing,
        )
        loss = out.loss / cfg.grad_accum_steps
        loss.backward()
        accum += 1

        if accum < cfg.grad_accum_steps:
            continue

        lr = linear_warmup_cosine(
            step,
            warmup_steps=cfg.warmup_steps,
            total_steps=cfg.total_steps,
            base_lr=cfg.base_lr,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr
        if cfg.grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                cfg.grad_clip_norm,
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        accum = 0

        stats = StepStats(
            step=step,
            loss=float(out.loss.detach()),
            loss_text=float(out.loss_text),
            loss_loc=float(out.loss_loc),
            lr=lr,
            n_text_tokens=out.n_text_tokens,
            n_loc_tokens=out.n_loc_tokens,
        )
        history.append(stats)
        if on_step is not None:
            on_step(stats)
        if step % cfg.log_every == 0:
            LOG.info(
                "step=%d loss=%.4f text=%.4f loc=%.4f lr=%.2e",
                step, stats.loss, stats.loss_text, stats.loss_loc, lr,
            )
        step += 1

    return history
