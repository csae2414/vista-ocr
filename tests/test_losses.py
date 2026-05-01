"""Combined-loss tests: term separation, prompt masking, gradient flow."""
from __future__ import annotations

import torch

from vista_ocr.training.losses import combined_loss


def _toy_batch(vocab_size: int = 20, t: int = 8, b: int = 2):
    torch.manual_seed(0)
    logits = torch.randn(b, t, vocab_size, requires_grad=True)
    labels = torch.randint(low=2, high=vocab_size, size=(b, t))
    return logits, labels


def test_loss_runs_with_no_prompt_mask():
    logits, labels = _toy_batch()
    out = combined_loss(logits, labels, spatial_token_ids={5, 6, 7}, lambda_text=0.5)
    assert out.loss.dim() == 0
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert logits.grad is not None


def test_loss_separates_text_vs_loc_token_counts():
    logits, labels = _toy_batch(vocab_size=20, t=10, b=1)
    # Force half the tokens to be spatial.
    labels[0, ::2] = 5
    out = combined_loss(logits, labels, spatial_token_ids={5})
    assert out.n_loc_tokens == 5
    assert out.n_text_tokens == 5


def test_prompt_mask_excludes_tokens():
    logits, labels = _toy_batch(t=6, b=1)
    prompt_mask = torch.zeros_like(labels, dtype=torch.bool)
    prompt_mask[0, :3] = True
    out_full = combined_loss(logits.detach().clone().requires_grad_(True), labels, {5})
    out_masked = combined_loss(
        logits.detach().clone().requires_grad_(True), labels, {5}, prompt_mask=prompt_mask
    )
    assert out_masked.n_text_tokens + out_masked.n_loc_tokens == 3
    assert out_full.n_text_tokens + out_full.n_loc_tokens == 6


def test_pad_tokens_ignored():
    logits, labels = _toy_batch(t=4, b=1)
    labels[0, :2] = 0  # pad
    out = combined_loss(logits, labels, {5}, pad_id=0)
    assert out.n_text_tokens + out.n_loc_tokens == 2


def test_zero_tokens_returns_finite_zero_grad():
    logits = torch.randn(1, 3, 5, requires_grad=True)
    labels = torch.zeros(1, 3, dtype=torch.long)  # all padding
    out = combined_loss(logits, labels, {2}, pad_id=0)
    assert torch.isfinite(out.loss)
    out.loss.backward()


def test_lambda_balance_extremes():
    logits, labels = _toy_batch()
    out_text = combined_loss(logits, labels, {5}, lambda_text=1.0)
    out_loc = combined_loss(logits, labels, {5}, lambda_text=0.0)
    # lambda=1 -> total == text term; lambda=0 -> total == loc term.
    assert torch.allclose(out_text.loss, out_text.loss_text)
    assert torch.allclose(out_loc.loss, out_loc.loss_loc)
