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
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from vista_ocr.data.collate import Batch, collate
from vista_ocr.data.preprocess import PreprocessConfig
from vista_ocr.data.types import Sample
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.tokenizer import VistaTokenizer
from vista_ocr.training.callbacks import (
    CheckpointConfig,
    EarlyStopConfig,
    EarlyStopState,
    ValConfig,
    early_stop_decision,
    find_latest_checkpoint,
    load_checkpoint,
    prune_old_checkpoints,
    run_validation,
    save_checkpoint,
)
from vista_ocr.training.losses import combined_loss
from vista_ocr.training.schedules import exponential_dropout, linear_warmup_cosine

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
    label_smoothing: float = 0.1                  # mBART finetuning standard
    pad_multiple: int = 32
    target_h: int = 3508
    target_w: int = 2480
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    freeze_decoder: bool = False
    # mBART-paper Adam betas/eps (vs (0.9, 0.999) / 1e-8 generic default).
    adam_betas: tuple[float, float] = (0.9, 0.98)
    adam_eps: float = 1e-6
    # Speed knobs for the GPU VM. Have no effect on CPU.
    gradient_checkpointing: bool = False
    autocast_dtype: torch.dtype | None = None     # set to torch.bfloat16 on A100
    compile_model: bool = False                    # torch.compile encoder+decoder
    # Persistence + validation. Disabled by default so the existing tests
    # still see a no-side-effect train(...).
    checkpoint: CheckpointConfig | None = None
    val: ValConfig | None = None
    val_batches_factory: object | None = None     # callable -> Iterable[Batch]
    val_loss_fn: object | None = None             # callable(model, batch) -> Tensor
    val_decode_fn: object | None = None           # callable(model, batch) -> (refs, hyps)
    val_decode_n: int = 0                         # >0 enables CER/WER on first N batches
    # Phase-2 DS-fix: every-val ``val_decode_n`` is small (5) so the
    # per-step log line is cheap. When a val pass is a ckpt_best
    # candidate (val_loss improved), fire a *second* eval pass with
    # ``val_decode_n_best`` samples (default 256) and persist the
    # bigger-sample CER/WER/word-F1 in the checkpoint's ``extra`` dict.
    # 0 disables the second pass and keeps the prior behaviour.
    val_decode_n_best: int = 0
    # Early stopping (Phase 8). Off by default; on per-stage when
    # operator opts in via ``--early-stop`` flag.
    early_stop: "EarlyStopConfig | None" = None
    resume_from: Path | None = None               # explicit path to a ckpt_*.pt
    # A4: cosine schedule floor. 0.0 = decay to zero (old behaviour);
    # 0.05 keeps a tiny LR through the tail. Documented as fresh-runs-only
    # in notes; mid-run change is not safe with checkpointed optimiser
    # state.
    min_lr_ratio: float = 0.0
    # B1: DANIEL exponential dropout schedule for the encoder.
    # Set encoder_dropout_max to None to disable (default off so no
    # behaviour change for existing tests). When enabled, per step:
    #   p(step) = encoder_dropout_max * (1 - exp(-step / dropout_T))
    # Stage-1 (frozen decoder, encoder-only) should leave this disabled
    # and use a fixed dropout to avoid early overfit.
    encoder_dropout_max: float | None = None
    dropout_T: float = 5e4


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
    """Build AdamW with mBART-style betas/eps and split weight-decay
    groups. Only parameters with ``requires_grad=True`` are included --
    important when ``freeze_decoder=True`` so frozen decoder weights are
    not allocated optimizer state."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim == 1 or n.endswith(".bias") else decay).append(p)
    LOG.info(
        "Optimizer: %d decay params, %d no_decay params (frozen excluded)",
        sum(p.numel() for p in decay),
        sum(p.numel() for p in no_decay),
    )
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.base_lr,
        betas=cfg.adam_betas,
        eps=cfg.adam_eps,
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


def _peek_first(it: Iterable) -> tuple[object, Iterator]:
    """Look at the first element without consuming the iterator."""
    inner = iter(it)
    first = next(inner)

    def _chain() -> Iterator:
        yield first
        yield from inner

    return first, _chain()


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
    # Freeze BEFORE optimizer construction so frozen params are excluded
    # from optimizer state (saves memory + avoids wasted update calls).
    if cfg.freeze_decoder:
        model.freeze_decoder(True)
    if cfg.gradient_checkpointing and hasattr(model.encoder, "enable_gradient_checkpointing"):
        model.encoder.enable_gradient_checkpointing(True)
    if cfg.compile_model:
        try:
            model = torch.compile(model)
            LOG.info("torch.compile() applied to model")
        except Exception as e:                    # noqa: BLE001
            LOG.warning("torch.compile failed (%s); continuing eagerly", e)
    optimizer = make_optimizer(model, cfg)
    spatial_ids = tokenizer._spatial_ids
    autocast_enabled = cfg.autocast_dtype is not None and device.type == "cuda"
    pre_cfg = PreprocessConfig(
        target_h=cfg.target_h, target_w=cfg.target_w, pad_multiple=cfg.pad_multiple
    )

    history: list[StepStats] = []
    accum = 0
    optimizer.zero_grad(set_to_none=True)
    step = 0
    best_val_loss = float("inf")
    early_stop_state = EarlyStopState()

    # Resume from checkpoint if requested or auto-discover the latest in
    # the configured checkpoint directory.
    resume_path: Path | None = cfg.resume_from
    if resume_path is None and cfg.checkpoint is not None:
        latest = find_latest_checkpoint(cfg.checkpoint.out_dir)
        if latest is not None:
            resume_path = latest
    if resume_path is not None:
        payload = load_checkpoint(
            resume_path, model=model, optimizer=optimizer, map_location=device
        )
        step = payload.step
        best_val_loss = payload.best_val_loss
        # F2: restore early-stop state so a resumed run keeps counting
        # patience from where the kill happened.
        early_stop_state = EarlyStopState.from_dict(
            payload.extra.get("early_stop_state"),
        )
        LOG.info("Resumed from %s at step %d (best_val_loss=%.4f)",
                 resume_path, step, best_val_loss)

    # Accept either a stream of Samples (collate inline -- simple, slow) or
    # a stream of pre-collated Batches (e.g. from
    # vista_ocr.data.dataloader.make_pdfa_dataloader -- multi-worker, fast).
    first, sample_stream = _peek_first(sample_stream)
    if isinstance(first, Batch):
        micro_iter = sample_stream
    elif isinstance(first, Sample):
        micro_iter = _iter_batches(
            sample_stream, tokenizer, pre_cfg, cfg.micro_batch_size
        )
    else:
        raise TypeError(
            f"sample_stream must yield Sample or Batch, got {type(first).__name__}"
        )

    for batch in micro_iter:
        if max_steps is not None and step >= max_steps:
            break

        # B1: DANIEL exponential dropout schedule for the encoder.
        # Per step we update the encoder's MixDropout p in place. No-op
        # when encoder_dropout_max is None.
        if cfg.encoder_dropout_max is not None and hasattr(
            model.encoder, "set_dropout"
        ):
            p = cfg.encoder_dropout_max * exponential_dropout(
                step, T=cfg.dropout_T
            )
            model.encoder.set_dropout(p)

        batch_device = Batch(
            images=batch.images.to(device),
            decoder_input_ids=batch.decoder_input_ids.to(device),
            labels=batch.labels.to(device),
            prompt_mask=batch.prompt_mask.to(device),
            pad_id=batch.pad_id,
        )

        if autocast_enabled:
            with torch.autocast(device_type=device.type, dtype=cfg.autocast_dtype):
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
        else:
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
            min_lr_ratio=cfg.min_lr_ratio,
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

        # Periodic validation
        if (
            cfg.val is not None
            and cfg.val_batches_factory is not None
            and cfg.val_loss_fn is not None
            and step > 0
            and step % cfg.val.every == 0
        ):
            # Free fragmented allocator pages -- without this, the val
            # forward (which runs without grad checkpointing materialised
            # activations) can hit transient memory pressure that surfaces
            # as CUBLAS_STATUS_EXECUTION_FAILED on a 24 GB card.
            if device.type == "cuda":
                torch.cuda.empty_cache()
            val_batches = cfg.val_batches_factory()
            val_stats = run_validation(
                model, val_batches, cfg.val_loss_fn,
                max_batches=cfg.val.max_batches,
                decode_fn=cfg.val_decode_fn,
                decode_n=cfg.val_decode_n,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            extras = ""
            if "val_cer" in val_stats:
                extras = (
                    f" cer={val_stats['val_cer']:.4f}"
                    f" wer={val_stats['val_wer']:.4f}"
                    f" word-f1={val_stats['val_word_f1']:.4f}"
                    f" decoded={val_stats['val_decoded_n']}"
                    f" empty={val_stats['val_decoded_n_empty']}"
                )
            LOG.info("validation step=%d val_loss=%.4f n=%d (%.1fs)%s",
                     step, val_stats["val_loss"], val_stats["n_batches"],
                     val_stats["elapsed_s"], extras)
            if cfg.checkpoint and cfg.checkpoint.save_best and \
                    val_stats["val_loss"] < best_val_loss:
                best_val_loss = val_stats["val_loss"]
                # Phase-2 DS-fix: when val_loss improvement marks this
                # checkpoint as a ckpt_best candidate, fire a second
                # eval pass with a much larger ``decode_n`` so the
                # ckpt_best metadata carries a low-noise CER / word-F1
                # number. The first pass kept ``decode_n`` small (=5)
                # to keep every val pass cheap; the candidate pass
                # runs only on improvement events (a handful per stage).
                ckpt_extra = {
                    "val_loss": val_stats["val_loss"],
                    "val_n_batches": val_stats["n_batches"],
                    "val_decoded_n": val_stats.get("val_decoded_n", 0),
                    "early_stop_state": early_stop_state.to_dict(),
                }
                if (
                    cfg.val_decode_n_best > 0
                    and cfg.val_decode_fn is not None
                    and cfg.val_batches_factory is not None
                ):
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    big_batches = cfg.val_batches_factory()
                    big_stats = run_validation(
                        model, big_batches, cfg.val_loss_fn,
                        max_batches=max(
                            cfg.val.max_batches, cfg.val_decode_n_best,
                        ),
                        decode_fn=cfg.val_decode_fn,
                        decode_n=cfg.val_decode_n_best,
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    LOG.info(
                        "BEST_CANDIDATE: step=%d val_loss=%.4f "
                        "cer=%.4f wer=%.4f word_f1=%.4f n=%d empty=%d",
                        step, big_stats["val_loss"],
                        big_stats.get("val_cer", float("nan")),
                        big_stats.get("val_wer", float("nan")),
                        big_stats.get("val_word_f1", float("nan")),
                        big_stats.get("val_decoded_n", 0),
                        big_stats.get("val_decoded_n_empty", 0),
                    )
                    ckpt_extra["best_candidate"] = {
                        "val_loss": big_stats["val_loss"],
                        "val_cer": big_stats.get("val_cer"),
                        "val_wer": big_stats.get("val_wer"),
                        "val_word_f1": big_stats.get("val_word_f1"),
                        "val_decoded_n": big_stats.get("val_decoded_n", 0),
                        "val_decoded_n_empty": big_stats.get(
                            "val_decoded_n_empty", 0,
                        ),
                    }
                save_checkpoint(
                    cfg.checkpoint.out_dir / "ckpt_best.pt",
                    step=step, model=model, optimizer=optimizer,
                    best_val_loss=best_val_loss,
                    extra=ckpt_extra,
                )

            # Phase 8: early-stop decision after every val pass.
            should_stop = False
            stop_reason: str | None = None
            if cfg.early_stop is not None and cfg.early_stop.enabled:
                should_stop, stop_reason = early_stop_decision(
                    val_loss=float(val_stats["val_loss"]),
                    state=early_stop_state,
                    cfg=cfg.early_stop,
                )
            if should_stop:
                # F6: structured log line so the tailer parses it as
                # an event, not a free-form INFO line.
                LOG.info(
                    "EARLY_STOP: step=%d reason=%s smoothed_best=%.4f "
                    "patience=%d no_improve=%d spike=%d",
                    step, stop_reason,
                    early_stop_state.smoothed_best,
                    cfg.early_stop.patience,
                    early_stop_state.no_improve_counter,
                    early_stop_state.spike_counter,
                )
                # Persist final ckpt with early-stop state so the next
                # stage's loader can read it for forensics if it wants.
                if cfg.checkpoint is not None and getattr(
                    cfg.checkpoint, "save_final", True,
                ):
                    save_checkpoint(
                        cfg.checkpoint.out_dir / "ckpt_final.pt",
                        step=step, model=model, optimizer=optimizer,
                        best_val_loss=best_val_loss,
                        extra={
                            "reason": "early_stop",
                            "early_stop_reason": stop_reason,
                            "early_stop_state": early_stop_state.to_dict(),
                        },
                    )
                return history

        # Periodic checkpoint
        if cfg.checkpoint is not None and step > 0 and step % cfg.checkpoint.save_every == 0:
            save_checkpoint(
                cfg.checkpoint.out_dir / f"ckpt_{step:08d}.pt",
                step=step, model=model, optimizer=optimizer,
                best_val_loss=best_val_loss,
                extra={"early_stop_state": early_stop_state.to_dict()},
            )
            prune_old_checkpoints(cfg.checkpoint.out_dir, cfg.checkpoint.keep_last)

        step += 1

    # End-of-loop: save ckpt_final.pt alongside ckpt_best.pt. Final
    # captures the last training step's state -- not val-loss-selected,
    # so downstream evals can compare both. ckpt_best may be earlier
    # if val_loss is a misleading signal for the run.
    if cfg.checkpoint is not None and getattr(cfg.checkpoint, "save_final", True):
        save_checkpoint(
            cfg.checkpoint.out_dir / "ckpt_final.pt",
            step=step, model=model, optimizer=optimizer,
            best_val_loss=best_val_loss,
            extra={"reason": "end_of_training"},
        )

    return history
