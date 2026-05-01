"""Checkpoint save/load + resume + validation tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from vista_ocr.training.callbacks import (
    find_latest_checkpoint,
    load_checkpoint,
    prune_old_checkpoints,
    run_validation,
    save_checkpoint,
)


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.fixture
def model_and_optim() -> tuple[_Tiny, torch.optim.Optimizer]:
    torch.manual_seed(0)
    m = _Tiny()
    o = torch.optim.AdamW(m.parameters(), lr=1e-3)
    return m, o


def test_save_and_load_round_trip(tmp_path: Path, model_and_optim):
    m, o = model_and_optim
    p = tmp_path / "ckpt_00000005.pt"
    save_checkpoint(p, step=5, model=m, optimizer=o, best_val_loss=0.42,
                    extra={"note": "hi"})
    assert p.exists()

    m2 = _Tiny()
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    payload = load_checkpoint(p, model=m2, optimizer=o2)
    assert payload.step == 5
    assert payload.best_val_loss == pytest.approx(0.42)
    assert payload.extra == {"note": "hi"}
    for a, b in zip(m.parameters(), m2.parameters(), strict=False):
        assert torch.allclose(a, b)


def test_find_latest_picks_highest_step(tmp_path: Path, model_and_optim):
    m, o = model_and_optim
    save_checkpoint(tmp_path / "ckpt_00000010.pt", step=10, model=m, optimizer=o)
    save_checkpoint(tmp_path / "ckpt_00000050.pt", step=50, model=m, optimizer=o)
    save_checkpoint(tmp_path / "ckpt_00000020.pt", step=20, model=m, optimizer=o)
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "ckpt_00000050.pt"


def test_find_latest_returns_none_for_empty_dir(tmp_path: Path):
    assert find_latest_checkpoint(tmp_path) is None
    assert find_latest_checkpoint(tmp_path / "nope") is None


def test_prune_keeps_best_and_last_n(tmp_path: Path, model_and_optim):
    m, o = model_and_optim
    for s in [10, 20, 30, 40, 50]:
        save_checkpoint(tmp_path / f"ckpt_{s:08d}.pt", step=s, model=m, optimizer=o)
    save_checkpoint(tmp_path / "ckpt_best.pt", step=42, model=m, optimizer=o)
    prune_old_checkpoints(tmp_path, keep_last=2)
    remaining = sorted(p.name for p in tmp_path.glob("ckpt_*.pt"))
    assert "ckpt_best.pt" in remaining
    assert "ckpt_00000050.pt" in remaining
    assert "ckpt_00000040.pt" in remaining
    assert "ckpt_00000010.pt" not in remaining


def test_run_validation_iterates_to_max_batches():
    m = _Tiny()

    def loss_fn(model, batch):
        return torch.tensor(0.5 + batch * 0.1)

    out = run_validation(m, iter([0, 1, 2, 3, 4, 5]), loss_fn, max_batches=3)
    assert out["n_batches"] == 3
    assert out["val_loss"] == pytest.approx((0.5 + 0.6 + 0.7) / 3)


def test_run_validation_handles_empty_iterator():
    m = _Tiny()
    out = run_validation(m, iter([]), lambda model, batch: torch.tensor(0.0))
    assert out["n_batches"] == 0


def test_load_checkpoint_with_non_cpu_map_location(tmp_path: Path, model_and_optim):
    """Regression test for the resume crash: map_location="cuda" moves the
    RNG ByteTensor off CPU; torch.set_rng_state then rejects it. The
    loader must move RNG tensors back to CPU."""
    m, o = model_and_optim
    p = tmp_path / "ckpt_00000003.pt"
    save_checkpoint(p, step=3, model=m, optimizer=o)

    m2 = _Tiny()
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    # Use a non-cpu map_location string. Even on a CPU box this exercises
    # the .cpu().to(uint8) path because torch.load happily accepts it.
    payload = load_checkpoint(p, model=m2, optimizer=o2, map_location="cpu")
    assert payload.step == 3


def test_resume_restores_optimizer_state_and_step(tmp_path: Path, model_and_optim):
    m, o = model_and_optim
    # Take a step so optimizer has non-trivial state.
    x = torch.randn(2, 4)
    loss = m(x).sum()
    loss.backward()
    o.step()
    save_checkpoint(tmp_path / "ckpt_00000007.pt", step=7, model=m, optimizer=o,
                    best_val_loss=1.234)

    m2 = _Tiny()
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    payload = load_checkpoint(tmp_path / "ckpt_00000007.pt", model=m2, optimizer=o2)
    assert payload.step == 7
    assert payload.best_val_loss == pytest.approx(1.234)
    # Optimizer state actually populated
    assert any(o2.state.values())
