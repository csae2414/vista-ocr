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
    metric_initial_best,
    metric_is_better,
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
    # DS-fix Phase 3: startup-time validation. ``select_on=val_word_f1``
    # only works when the val pass actually computes word_f1 -- i.e.
    # when val_decode_fn is set AND val_decode_n > 0. Without that
    # the gate value is None on every val pass and ckpt_best.pt is
    # never written. Hard-fail at startup (not silently 8 h later).
    if (
        cfg.checkpoint is not None
        and cfg.checkpoint.save_best
        and cfg.checkpoint.select_on == "val_word_f1"
        and cfg.val is not None
        and (cfg.val_decode_fn is None or cfg.val_decode_n <= 0)
    ):
        raise ValueError(
            "checkpoint.select_on='val_word_f1' but val_decode_fn / "
            "val_decode_n are not configured to compute word_f1. "
            "Set --decode-n > 0 (and ensure val_decode_fn is wired) "
            "or pass --select-on val_loss."
        )
    if (
        cfg.early_stop is not None
        and cfg.early_stop.enabled
        and cfg.early_stop.metric == "val_word_f1"
        and cfg.val is not None
        and (cfg.val_decode_fn is None or cfg.val_decode_n <= 0)
    ):
        raise ValueError(
            "early_stop.metric='val_word_f1' but val_decode_fn / "
            "val_decode_n are not configured to compute word_f1."
        )
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
    select_metric = (
        cfg.checkpoint.select_on if cfg.checkpoint is not None else "val_loss"
    )
    best_metric = metric_initial_best(select_metric)
    # Tracked independently of ``select_metric`` so periodic + final
    # ckpts always carry "best val_loss seen" in their top-level field
    # for back-compat with downstream readers.
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
        # Resume back-compat: older ckpts stored only ``best_val_loss``.
        # When the new selection metric is val_word_f1, prefer the
        # explicit ``best_metric`` if the ckpt has it; otherwise reset
        # to the initial best (the prior best_val_loss is on a
        # different scale and must not be reused).
        ckpt_best_metric = payload.extra.get("best_metric")
        ckpt_best_metric_name = payload.extra.get("best_metric_name")
        if (
            ckpt_best_metric is not None
            and ckpt_best_metric_name == select_metric
        ):
            best_metric = float(ckpt_best_metric)
        elif select_metric == "val_loss":
            best_metric = payload.best_val_loss
        # Always restore best_val_loss for the periodic/final ckpt
        # top-level field; meaningful regardless of select_metric.
        best_val_loss = payload.best_val_loss
        # F2: restore early-stop state so a resumed run keeps counting
        # patience from where the kill happened.
        early_stop_state = EarlyStopState.from_dict(
            payload.extra.get("early_stop_state"),
        )
        LOG.info(
            "Resumed from %s at step %d (best_%s=%.4f)",
            resume_path, step, select_metric, best_metric,
        )

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
            if val_stats["val_loss"] < best_val_loss:
                best_val_loss = float(val_stats["val_loss"])
            # DS-fix Phase 3: two-stage selection.
            #   1. GATE on the cheap (n=val_decode_n) signal: skip the
            #      expensive second pass when the cheap metric clearly
            #      hasn't improved.
            #   2. CONFIRM with the second-pass (n=val_decode_n_best,
            #      typically 256). If the second-pass metric improves
            #      over ``best_metric`` we save ckpt_best and ratchet
            #      ``best_metric``. If it doesn't (the gate fired on
            #      noise), we skip the save without ratcheting -- so
            #      a future *real* improvement is not locked out by
            #      a noise spike on n=5.
            # When val_decode_n_best == 0, the second pass is disabled
            # and the gate IS the truth (legacy single-pass behaviour).
            gate_value = val_stats.get(select_metric)
            gate_improved = (
                gate_value is not None
                and cfg.checkpoint is not None
                and cfg.checkpoint.save_best
                and metric_is_better(gate_value, best_metric, select_metric)
            )
            if gate_improved:
                run_second_pass = (
                    cfg.val_decode_n_best > 0
                    and cfg.val_decode_fn is not None
                    and cfg.val_batches_factory is not None
                )
                truth_value = gate_value
                truth_stats: dict | None = None
                if run_second_pass:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    big_batches = cfg.val_batches_factory()
                    truth_stats = run_validation(
                        model, big_batches, cfg.val_loss_fn,
                        max_batches=max(
                            cfg.val.max_batches, cfg.val_decode_n_best,
                        ),
                        decode_fn=cfg.val_decode_fn,
                        decode_n=cfg.val_decode_n_best,
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    truth_value = truth_stats.get(select_metric, gate_value)
                    LOG.info(
                        "BEST_CANDIDATE: step=%d selected=%s=%.4f "
                        "val_loss=%.4f cer=%.4f wer=%.4f word_f1=%.4f "
                        "n=%d empty=%d",
                        step, select_metric, float(truth_value),
                        truth_stats["val_loss"],
                        truth_stats.get("val_cer", float("nan")),
                        truth_stats.get("val_wer", float("nan")),
                        truth_stats.get("val_word_f1", float("nan")),
                        truth_stats.get("val_decoded_n", 0),
                        truth_stats.get("val_decoded_n_empty", 0),
                    )
                # Truth-pass confirmation: only save and ratchet if the
                # truth_value still beats best_metric. This is the noise
                # filter -- without it, a noisy gate spike both writes
                # ckpt_best AND raises best_metric, locking out future
                # real improvements.
                if metric_is_better(truth_value, best_metric, select_metric):
                    best_metric = float(truth_value)
                    ckpt_extra = {
                        "val_loss": val_stats["val_loss"],
                        "val_n_batches": val_stats["n_batches"],
                        "val_decoded_n": val_stats.get("val_decoded_n", 0),
                        "best_metric": best_metric,
                        "best_metric_name": select_metric,
                        "early_stop_state": early_stop_state.to_dict(),
                    }
                    if truth_stats is not None:
                        ckpt_extra["best_candidate"] = {
                            "val_loss": truth_stats["val_loss"],
                            "val_cer": truth_stats.get("val_cer"),
                            "val_wer": truth_stats.get("val_wer"),
                            "val_word_f1": truth_stats.get("val_word_f1"),
                            "val_decoded_n": truth_stats.get(
                                "val_decoded_n", 0,
                            ),
                            "val_decoded_n_empty": truth_stats.get(
                                "val_decoded_n_empty", 0,
                            ),
                        }
                    save_checkpoint(
                        cfg.checkpoint.out_dir / "ckpt_best.pt",
                        step=step, model=model, optimizer=optimizer,
                        best_val_loss=float(val_stats["val_loss"]),
                        extra=ckpt_extra,
                    )
                else:
                    LOG.info(
                        "BEST_CANDIDATE_REJECTED: step=%d gate=%s=%.4f "
                        "truth=%s=%.4f best=%.4f -- noise filter",
                        step, select_metric, float(gate_value),
                        select_metric, float(truth_value), best_metric,
                    )

            # Phase 8: early-stop decision after every val pass.
            should_stop = False
            stop_reason: str | None = None
            if cfg.early_stop is not None and cfg.early_stop.enabled:
                # DS-fix P3: early-stop uses the same metric as ckpt_best
                # (cfg.early_stop.metric, default val_word_f1). The
                # decision function takes the value + a higher_is_better
                # flag so EMA / patience / spike directions stay correct.
                es_metric = cfg.early_stop.metric
                es_value = val_stats.get(es_metric)
                if es_value is None:
                    # If the configured metric isn't present (e.g.
                    # val_word_f1 with decode_n=0), fall back to val_loss.
                    es_metric = "val_loss"
                    es_value = val_stats["val_loss"]
                should_stop, stop_reason = early_stop_decision(
                    val_loss=float(es_value),
                    state=early_stop_state,
                    cfg=cfg.early_stop,
                )
            if should_stop:
                # F6: structured log line so the tailer parses it as
                # an event, not a free-form INFO line.
                LOG.info(
                    "EARLY_STOP: step=%d reason=%s metric=%s smoothed_best=%.4f "
                    "patience=%d no_improve=%d spike=%d",
                    step, stop_reason, es_metric,
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
