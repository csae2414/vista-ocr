"""Training callbacks: checkpoint save/load and periodic validation.

Lightweight building blocks the training loop calls at fixed step
intervals. They live outside the loop so unit tests can exercise them
in isolation.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from torch import nn

LOG = logging.getLogger(__name__)


SelectMetric = Literal["val_loss", "val_word_f1"]


def metric_is_better(value: float, best: float, metric: SelectMetric) -> bool:
    """Direction-aware comparison for ``ckpt_best`` and early-stop.

    val_loss is lower-is-better; val_word_f1 is higher-is-better.
    Centralised so checkpoint selection and early-stop can't disagree
    on direction.
    """
    if metric == "val_loss":
        return value < best
    if metric == "val_word_f1":
        return value > best
    raise ValueError(f"unknown selection metric: {metric!r}")


def metric_initial_best(metric: SelectMetric) -> float:
    """Initial ``best`` value such that the first observation always
    counts as an improvement."""
    if metric == "val_loss":
        return float("inf")
    if metric == "val_word_f1":
        return float("-inf")
    raise ValueError(f"unknown selection metric: {metric!r}")


@dataclass
class CheckpointConfig:
    out_dir: Path
    save_every: int = 5000
    keep_last: int = 3
    save_best: bool = True
    # When True, also write ``ckpt_final.pt`` at the end of the training
    # loop alongside ``ckpt_best.pt``. ``ckpt_best.pt`` is metric-
    # selected (see ``select_on``); the final ckpt captures whatever
    # state training ended at, which downstream evals may want for
    # sanity checks.
    save_final: bool = True
    # DS-fix Phase 3: ckpt_best selection criterion. The 2026-05-02
    # PDFA hold-out diagnosis showed val_loss can be misleading when
    # the model has weak image conditioning -- the language-model
    # prior dominates so val_loss reflects the prior, not OCR
    # quality. word-F1 from the second-pass eval (``val_decode_n_best``,
    # default 256 in stage scripts) is the recommended criterion.
    # ``val_loss`` is kept for tests / back-compat with old configs.
    select_on: SelectMetric = "val_word_f1"


@dataclass
class EarlyStopConfig:
    """Production-ready early stopping for the val-loss curve.

    Disabled by default. When enabled, the training loop returns
    cleanly (exit code 0) once the smoothed val-loss has not improved
    for ``patience`` consecutive val passes, OR the raw val-loss has
    spiked above the smoothed-best by more than ``spike_threshold``
    for ``spike_consecutive`` consecutive passes.

    The state ``(no_improve_counter, smoothed_best, val_history,
    spike_counter)`` is persisted into the checkpoint's ``extra``
    dict so a resumed run picks up where the prior counter left off.

    Two early-exit reasons:

    * ``patience_exceeded``: the smoothed val curve flattened.
    * ``spike_detected``: the raw val curve degraded sharply
      relative to the smoothed-best.

    Per-stage tuning knobs are exposed in stage scripts; sensible
    defaults differ between calibration (high tolerance, decoder
    frozen → asymptotic floor) and the unfrozen stages.
    """

    enabled: bool = False
    # N consecutive val passes with no smoothed improvement -> abort.
    patience: int = 10
    # Smaller smoothed delta than this counts as "no improvement".
    min_delta: float = 0.01
    # EMA window for the smoothed metric.
    smooth_window: int = 5
    # Don't start counting "no improvement" until this many vals have
    # accumulated. Avoids premature abort from the first noisy val
    # passes after a stage transition.
    warmup_vals: int = 5
    # Spike-detection threshold (degradation that exceeds the
    # smoothed-best by this much for ``spike_consecutive`` vals -> abort).
    # For ``metric=val_loss`` "exceed" means *higher* than smoothed_best;
    # for ``metric=val_word_f1`` it means *lower*. Catches a fast collapse
    # the smoothed signal would otherwise mask.
    spike_threshold: float = 0.5
    spike_consecutive: int = 3
    # DS-fix Phase 3: which metric to watch. Defaults to val_word_f1 to
    # match CheckpointConfig.select_on. ``val_loss`` is kept for
    # back-compat / tests where the cheap loss is the only signal.
    metric: SelectMetric = "val_word_f1"


@dataclass
class EarlyStopState:
    """Mutable state the train loop carries across val passes.

    Persisted into the checkpoint's ``extra`` dict so a resume
    restores patience counting correctly. F2 in the design notes:
    without this, a kill+restart loses the patience counter and we
    pay another full ``patience * val_every`` steps before aborting.
    """

    val_history: list[float] = field(default_factory=list)
    smoothed_best: float = float("inf")
    no_improve_counter: int = 0
    spike_counter: int = 0
    n_vals_seen: int = 0

    def to_dict(self) -> dict:
        return {
            "val_history": list(self.val_history),
            "smoothed_best": self.smoothed_best,
            "no_improve_counter": self.no_improve_counter,
            "spike_counter": self.spike_counter,
            "n_vals_seen": self.n_vals_seen,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> EarlyStopState:
        if not d:
            return cls()
        return cls(
            val_history=list(d.get("val_history", [])),
            smoothed_best=float(d.get("smoothed_best", float("inf"))),
            no_improve_counter=int(d.get("no_improve_counter", 0)),
            spike_counter=int(d.get("spike_counter", 0)),
            n_vals_seen=int(d.get("n_vals_seen", 0)),
        )


def _ema(values: list[float], window: int) -> float:
    """Exponential moving average over the last ``window`` values."""
    if not values:
        return float("inf")
    tail = values[-window:]
    if len(tail) == 1:
        return tail[0]
    alpha = 2.0 / (len(tail) + 1)
    smoothed = tail[0]
    for v in tail[1:]:
        smoothed = alpha * v + (1 - alpha) * smoothed
    return smoothed


def early_stop_decision(
    val_loss: float,
    state: EarlyStopState,
    cfg: EarlyStopConfig,
) -> tuple[bool, str | None]:
    """Update ``state`` in place with a new metric value; return
    ``(should_stop, reason)``.

    The first parameter is named ``val_loss`` for back-compat -- in
    practice it's whichever metric ``cfg.metric`` selects (val_loss or
    val_word_f1). Direction is taken from ``cfg.metric``: lower-is-
    better for val_loss, higher-is-better for val_word_f1.

    Reasons:
      * ``"patience_exceeded"`` -- smoothed best didn't improve for
        ``cfg.patience`` consecutive vals.
      * ``"spike_detected"`` -- raw value degraded past
        ``smoothed_best ± cfg.spike_threshold`` (sign chosen per
        direction) for ``cfg.spike_consecutive`` vals.

    The check is always applied AFTER ``state`` is updated, but never
    fires before ``cfg.warmup_vals`` vals have been observed.
    """
    state.val_history.append(val_loss)
    state.n_vals_seen += 1

    higher_is_better = cfg.metric == "val_word_f1"
    smoothed = _ema(state.val_history, cfg.smooth_window)

    # First val ever observed: bootstrap smoothed_best regardless of
    # direction so the higher-is-better init (-inf in spirit) and the
    # legacy lower-is-better init (inf) both work without special
    # cases at construction time.
    if state.n_vals_seen == 1:
        state.smoothed_best = smoothed
        state.no_improve_counter = 0
    else:
        if higher_is_better:
            improved = smoothed - cfg.min_delta > state.smoothed_best
        else:
            improved = smoothed + cfg.min_delta < state.smoothed_best
        if improved:
            state.smoothed_best = smoothed
            state.no_improve_counter = 0
        else:
            state.no_improve_counter += 1

    # Spike detection on RAW (not smoothed) value, direction-aware.
    if higher_is_better:
        spiked = val_loss < state.smoothed_best - cfg.spike_threshold
    else:
        spiked = val_loss > state.smoothed_best + cfg.spike_threshold
    if spiked:
        state.spike_counter += 1
    else:
        state.spike_counter = 0

    # Don't fire during warmup. ``warmup_vals=N`` means: the first N
    # vals never trigger abort. Abort is allowed from call N+1 onward.
    if state.n_vals_seen <= cfg.warmup_vals:
        return False, None
    if state.no_improve_counter >= cfg.patience:
        return True, "patience_exceeded"
    if state.spike_counter >= cfg.spike_consecutive:
        return True, "spike_detected"
    return False, None


@dataclass
class CheckpointPayload:
    """Everything we need to resume a training run."""

    step: int
    model_state: dict
    optimizer_state: dict
    rng_state_cpu: torch.Tensor
    rng_state_cuda: list[torch.Tensor]
    best_val_loss: float
    extra: dict = field(default_factory=dict)


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float = float("inf"),
    extra: dict | None = None,
) -> None:
    """Persist model + optimizer + RNG to ``path`` atomically (write to
    .tmp then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "best_val_loss": best_val_loss,
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    LOG.info("Checkpoint saved: %s (step %d, %.1f MB)", path, step, path.stat().st_size / 1e6)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> CheckpointPayload:
    """Load a checkpoint into ``model`` (and optionally ``optimizer``).
    Returns the parsed payload so the trainer can resume bookkeeping."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        # ``map_location`` may have moved the RNG tensors onto CUDA. Both
        # ``torch.set_rng_state`` and ``torch.cuda.set_rng_state_all``
        # require CPU ByteTensors, so we force them back here.
        cpu_state = payload["rng_cpu"].cpu().to(torch.uint8)
        torch.set_rng_state(cpu_state)
        if torch.cuda.is_available() and payload.get("rng_cuda"):
            cuda_states = [
                t.cpu().to(torch.uint8) for t in payload["rng_cuda"]
            ]
            torch.cuda.set_rng_state_all(cuda_states)
    LOG.info("Checkpoint loaded: %s (resumed from step %d)", path, payload["step"])
    return CheckpointPayload(
        step=int(payload["step"]),
        model_state=payload["model"],
        optimizer_state=payload.get("optimizer", {}),
        rng_state_cpu=payload["rng_cpu"],
        rng_state_cuda=payload.get("rng_cuda", []),
        best_val_loss=float(payload.get("best_val_loss", float("inf"))),
        extra=payload.get("extra", {}),
    )


def find_latest_checkpoint(out_dir: Path) -> Path | None:
    """Return the highest-step ``ckpt_*.pt`` file in ``out_dir``, or None."""
    if not out_dir.exists():
        return None
    candidates = []
    for p in out_dir.glob("ckpt_*.pt"):
        try:
            step = int(p.stem.split("_")[1])
        except (ValueError, IndexError):
            continue
        candidates.append((step, p))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def prune_old_checkpoints(out_dir: Path, keep_last: int) -> None:
    """Delete all but the most recent ``keep_last`` ``ckpt_*.pt`` files.
    Never deletes ``ckpt_best.pt``."""
    files = sorted(
        out_dir.glob("ckpt_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else -1,
    )
    files = [f for f in files if f.name != "ckpt_best.pt"]
    for old in files[:-keep_last]:
        old.unlink(missing_ok=True)
        LOG.info("Pruned old checkpoint: %s", old)


# ---------- validation ------------------------------------------------------

@dataclass
class ValConfig:
    every: int = 5000
    max_batches: int = 50


def run_validation(
    model: nn.Module,
    val_batches: Iterable,
    loss_fn: Callable[[nn.Module, object], torch.Tensor],
    *,
    max_batches: int = 50,
    decode_fn: Callable[[nn.Module, object], tuple[list[str], list[str]]] | None = None,
    decode_n: int = 0,
) -> dict:
    """Compute mean loss over the first ``max_batches`` of ``val_batches``.

    * ``loss_fn`` takes ``(model, batch)`` and returns a scalar tensor.
    * ``decode_fn``, when given, runs on the first ``decode_n`` batches
      and must return ``(refs, hyps)`` text lists. The CER/WER over
      those refs/hyps are added to the returned dict (``val_cer``,
      ``val_wer``, ``val_word_f1``, ``val_decoded_n_empty``).
    * Gradient checkpointing stays **on** during validation. Under
      ``torch.no_grad()`` the recomputation overhead is irrelevant and
      the materialised-activation peak memory would otherwise be ~3x
      higher than training.
    """
    was_training = model.training
    model.eval()
    losses: list[float] = []
    t0 = time.perf_counter()
    skipped = 0
    refs_for_decode: list[str] = []
    hyps_for_decode: list[str] = []
    # Force-disable autocast for the entire val pass. Issue #132613 in
    # PyTorch + cuBLAS internal allocation contention have been observed
    # to surface as CUBLAS_STATUS_EXECUTION_FAILED in eval mode + bf16
    # autocast on the MBart eager attention path. fp32 val is correct
    # and slow-but-rare (val runs every val.every steps).
    with torch.no_grad(), torch.autocast(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32, enabled=False,
    ):
        for i, batch in enumerate(val_batches):
            if i >= max_batches:
                break
            try:
                losses.append(float(loss_fn(model, batch)))
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                skipped += 1
                LOG.warning("val batch %d skipped: %s", i, exc)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            if decode_fn is not None and i < decode_n:
                try:
                    refs, hyps = decode_fn(model, batch)
                    refs_for_decode.extend(refs)
                    hyps_for_decode.extend(hyps)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    LOG.warning("val decode %d skipped: %s", i, exc)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    if was_training:
        model.train()

    decode_metrics: dict = {}
    if refs_for_decode:
        # Imported lazily so the loss-only path doesn't pay for jiwer.
        from vista_ocr.eval.metrics_recognition import recognition_metrics

        rec = recognition_metrics(refs_for_decode, hyps_for_decode)
        n_empty = sum(1 for h in hyps_for_decode if not h.strip())
        decode_metrics = {
            "val_cer": rec.cer,
            "val_wer": rec.wer,
            "val_word_f1": rec.f1,
            "val_decoded_n": len(hyps_for_decode),
            "val_decoded_n_empty": n_empty,
        }
        if n_empty > len(hyps_for_decode) // 2:
            LOG.warning(
                "val: %d/%d decoded outputs are empty -- decoder may be "
                "stuck (pad==eos regression?)", n_empty, len(hyps_for_decode),
            )

    if not losses:
        out = {"val_loss": float("nan"), "n_batches": 0, "elapsed_s": 0.0,
               "skipped": skipped}
    else:
        out = {
            "val_loss": sum(losses) / len(losses),
            "n_batches": len(losses),
            "elapsed_s": time.perf_counter() - t0,
            "skipped": skipped,
        }
    out.update(decode_metrics)
    return out
