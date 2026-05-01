"""Training callbacks: checkpoint save/load and periodic validation.

Lightweight building blocks the training loop calls at fixed step
intervals. They live outside the loop so unit tests can exercise them
in isolation.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch import nn

LOG = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    out_dir: Path
    save_every: int = 5000
    keep_last: int = 3
    save_best: bool = True


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
) -> dict:
    """Compute mean loss over the first ``max_batches`` of ``val_batches``.

    * ``loss_fn`` takes ``(model, batch)`` and returns a scalar tensor.
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
    if was_training:
        model.train()
    if not losses:
        return {"val_loss": float("nan"), "n_batches": 0, "elapsed_s": 0.0,
                "skipped": skipped}
    return {
        "val_loss": sum(losses) / len(losses),
        "n_batches": len(losses),
        "elapsed_s": time.perf_counter() - t0,
        "skipped": skipped,
    }
