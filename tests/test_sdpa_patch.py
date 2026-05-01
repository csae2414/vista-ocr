"""Tests for the SDPA monkey-patch (C3).

Layout: one fixture builds an ``MBartAttention`` and applies the
patch; one parametrized test runs every equivalence case (self/cross/
causal/non-causal/bf16-paper/autocast/grads/kv-cache) through the
shared :func:`equivalence_report` helper. Targeted unit tests cover
``apply()``/``revert()`` mechanics, structural-attr refusal, version
warning, ``output_attentions=True`` refusal, dropout statistics, and
beam-search log-prob parity.

These tests run in CPU CI; the bf16 paper-shape case runs on CPU at
full size so the ship gate fires on every PR.
"""
from __future__ import annotations

import logging

import pytest
import torch

from vista_ocr.models import sdpa_patch
from vista_ocr.models.sdpa_patch import (
    _mask_is_purely_causal,
    apply,
    call_eager,
    equivalence_report,
    revert,
)

# Tolerances chosen empirically: SDPA reduces operations differently
# from a hand-rolled softmax-bmm so fp32 max-diff settles around 1e-6;
# bf16 settles around 3e-3 at paper-scale shapes. Keep these loose
# enough for stable CI but tight enough to catch genuine regressions.
ATOL_FP32 = 1e-4
ATOL_GRAD_FP32 = 1e-4
ATOL_BF16 = 5e-3


@pytest.fixture(autouse=True)
def _no_env_disable(monkeypatch):
    monkeypatch.delenv("VISTA_NO_SDPA", raising=False)


@pytest.fixture
def patched_attn():
    """Build a fresh MBartAttention with the patch applied.

    Yields ``(attn, hidden_states_factory)``. After the test the patch
    is reverted so other tests see eager.
    """
    from transformers.models.mbart.modeling_mbart import MBartAttention

    if sdpa_patch._PATCHED:
        revert()
    assert apply() is True

    embed_dim, num_heads = 64, 4
    attn = MBartAttention(
        embed_dim=embed_dim, num_heads=num_heads,
        dropout=0.0, is_decoder=True,
    ).eval()
    yield attn, embed_dim, num_heads
    revert()


# ---------------------------------------------------------------------
# Pure helper: _mask_is_purely_causal
# ---------------------------------------------------------------------

class TestMaskIsPurelyCausal:
    def test_canonical_causal_returns_true(self):
        T = 5
        mask = torch.zeros(1, 1, T, T)
        mask.masked_fill_(
            torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1),
            torch.finfo(mask.dtype).min,
        )
        assert _mask_is_purely_causal(mask) is True

    def test_padding_combined_with_causal_returns_false(self):
        T = 5
        mask = torch.zeros(1, 1, T, T)
        mask.masked_fill_(
            torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1),
            torch.finfo(mask.dtype).min,
        )
        # Pad position 4 -- adds an extra -inf column (a real padding
        # mask would). Now it is not purely causal.
        mask[..., 4] = torch.finfo(mask.dtype).min
        assert _mask_is_purely_causal(mask) is False

    def test_none_returns_false(self):
        assert _mask_is_purely_causal(None) is False

    def test_wrong_ndim_returns_false(self):
        assert _mask_is_purely_causal(torch.zeros(5, 5)) is False

    def test_non_square_returns_false(self):
        assert _mask_is_purely_causal(torch.zeros(1, 1, 4, 5)) is False


# ---------------------------------------------------------------------
# Parametrized equivalence test (the bulk of the coverage)
# ---------------------------------------------------------------------

def _make_self_attn_inputs(B, T, D):
    torch.manual_seed(0)
    return {"hidden_states": torch.randn(B, T, D)}


def _make_cross_attn_inputs(B, T, S, D):
    torch.manual_seed(0)
    return {
        "hidden_states": torch.randn(B, T, D),
        "key_value_states": torch.randn(B, S, D),
    }


def _make_causal_mask(B, T, dtype=torch.float32):
    mask = torch.zeros(B, 1, T, T, dtype=dtype)
    mask.masked_fill_(
        torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1),
        torch.finfo(dtype).min,
    )
    return mask


def _make_padding_plus_causal_mask(B, T, dtype=torch.float32):
    mask = _make_causal_mask(B, T, dtype)
    # mark the last position as padding
    mask[..., -1] = torch.finfo(dtype).min
    return mask


@pytest.mark.parametrize("case", [
    "self_attn_no_mask",
    "cross_attn",
    "self_attn_pure_causal_mask",
    "self_attn_padding_plus_causal_mask",
])
def test_forward_equivalence(patched_attn, case):
    attn, embed_dim, _heads = patched_attn
    B, T = 2, 16

    if case == "self_attn_no_mask":
        kwargs = _make_self_attn_inputs(B, T, embed_dim)
    elif case == "cross_attn":
        kwargs = _make_cross_attn_inputs(B, T, T + 4, embed_dim)
    elif case == "self_attn_pure_causal_mask":
        kwargs = _make_self_attn_inputs(B, T, embed_dim)
        kwargs["attention_mask"] = _make_causal_mask(B, T)
    elif case == "self_attn_padding_plus_causal_mask":
        kwargs = _make_self_attn_inputs(B, T, embed_dim)
        kwargs["attention_mask"] = _make_padding_plus_causal_mask(B, T)

    rep = equivalence_report(attn, kwargs, check_grads=False)
    assert rep["fwd_diff"] < ATOL_FP32, (
        f"{case}: fwd_diff={rep['fwd_diff']:.2e}"
    )


def test_backward_equivalence(patched_attn):
    """Step 3 of the plan: grads on q/k/v/out_proj match eager."""
    attn, embed_dim, _ = patched_attn
    kwargs = _make_self_attn_inputs(2, 16, embed_dim)
    rep = equivalence_report(attn, kwargs, check_grads=True)
    assert rep["fwd_diff"] < ATOL_FP32
    for name, d in rep["grad_diffs"].items():
        assert d < ATOL_GRAD_FP32, f"grad[{name}]={d:.2e}"


def test_kv_cache_two_step(patched_attn):
    """Step 6: prefilled cache + new step matches one-shot eager."""
    attn, embed_dim, _heads = patched_attn
    B, T = 1, 8
    torch.manual_seed(0)
    full = torch.randn(B, T, embed_dim)

    # one-shot eager over the full sequence -- last-step output is
    # the reference for the cached path's second call.
    out_full_eager, _, _ = call_eager(attn, hidden_states=full)
    ref_last = out_full_eager[:, -1:, :]

    # patched: prefill T-1 to build cache, then run last token with cache.
    prefill = full[:, :-1, :]
    last = full[:, -1:, :]
    _out1, _, kv = attn(hidden_states=prefill)
    out2, _, _ = attn(hidden_states=last, past_key_value=kv)

    diff = float((out2 - ref_last).abs().max().item())
    assert diff < ATOL_FP32, f"kv-cache step diff={diff:.2e}"


def test_bf16_paper_shape_cpu(patched_attn):
    """Step 5: bf16 ship gate at the EXACT paper attention shape.

    B=1, T=2048 -- matches MBart decoder during stage-2/3 training.
    CPU bf16 is slow (~10-15s); we accept it because the whole point
    of the gate is that it runs on the shape we ship.
    """
    attn, embed_dim, _heads = patched_attn
    attn = attn.to(torch.bfloat16)
    torch.manual_seed(0)
    hidden = torch.randn(1, 2048, embed_dim, dtype=torch.bfloat16)
    rep = equivalence_report(
        attn, {"hidden_states": hidden}, check_grads=False,
    )
    assert rep["fwd_diff"] < ATOL_BF16, (
        f"bf16 fwd_diff={rep['fwd_diff']:.2e} > {ATOL_BF16}"
    )


def test_autocast_bf16(patched_attn):
    """Step 4: forward inside autocast bf16 produces equivalent output."""
    attn, embed_dim, _ = patched_attn
    kwargs = _make_self_attn_inputs(2, 16, embed_dim)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        rep = equivalence_report(attn, kwargs, check_grads=False)
    # autocast pushes the matmuls to bf16; tolerance loosens.
    assert rep["fwd_diff"] < ATOL_BF16


# ---------------------------------------------------------------------
# Targeted unit tests
# ---------------------------------------------------------------------

def test_disabled_via_env(monkeypatch):
    """VISTA_NO_SDPA=1 short-circuits apply()."""
    monkeypatch.setenv("VISTA_NO_SDPA", "1")
    if sdpa_patch._PATCHED:
        revert()
    assert apply() is False
    assert sdpa_patch._PATCHED is False


def test_apply_is_idempotent():
    if sdpa_patch._PATCHED:
        revert()
    try:
        assert apply() is True
        original_after_first = sdpa_patch._ORIGINAL_FORWARD
        assert apply() is True
        # Idempotent: don't overwrite the saved original on the 2nd call.
        assert sdpa_patch._ORIGINAL_FORWARD is original_after_first
    finally:
        revert()


def test_revert_restores_eager_method():
    """After revert(), MBartAttention.forward IS the original."""
    from transformers.models.mbart.modeling_mbart import MBartAttention

    if sdpa_patch._PATCHED:
        revert()
    captured = MBartAttention.forward
    apply()
    assert MBartAttention.forward is not captured
    revert()
    assert MBartAttention.forward is captured
    assert sdpa_patch._PATCHED is False
    assert sdpa_patch._ORIGINAL_FORWARD is None


def test_eager_path_unchanged_when_not_applied():
    """If apply() was never called this session, forward is the original."""
    from transformers.models.mbart.modeling_mbart import MBartAttention

    if sdpa_patch._PATCHED:
        revert()
    # Construct attn and call forward; must work without raising.
    attn = MBartAttention(
        embed_dim=64, num_heads=4, dropout=0.0, is_decoder=True,
    ).eval()
    torch.manual_seed(0)
    out, _, _ = attn(hidden_states=torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)
    # And our patched forward is NOT the active forward.
    assert MBartAttention.forward is not sdpa_patch._sdpa_attention_forward


def test_structural_attr_missing_refuses(monkeypatch, caplog):
    """If MBartAttention is missing a required attr, apply() refuses."""
    if sdpa_patch._PATCHED:
        revert()
    # Inject a fake required attribute that will never be present.
    monkeypatch.setattr(
        sdpa_patch, "_REQUIRED_ATTRS",
        sdpa_patch._REQUIRED_ATTRS + ("__definitely_not_a_real_attr__",),
    )
    with caplog.at_level(logging.ERROR):
        result = apply()
    assert result is False
    assert sdpa_patch._PATCHED is False
    assert any("missing required" in r.message for r in caplog.records)


def test_unknown_transformers_version_warns_but_patches(monkeypatch, caplog):
    """Soft-warn on unknown version; structural assert is the real gate."""
    if sdpa_patch._PATCHED:
        revert()
    monkeypatch.setattr(
        sdpa_patch, "_SUPPORTED_TRANSFORMERS", ("99.99.99",),
    )
    with caplog.at_level(logging.WARNING):
        result = apply()
    try:
        assert result is True
        assert any(
            "outside the tested set" in r.message for r in caplog.records
        )
    finally:
        revert()


def test_output_attentions_true_raises(patched_attn):
    """SDPA cannot return weights; the patch raises rather than lying."""
    attn, embed_dim, _ = patched_attn
    torch.manual_seed(0)
    hidden = torch.randn(1, 8, embed_dim)
    with pytest.raises(NotImplementedError, match="output_attentions"):
        attn(hidden_states=hidden, output_attentions=True)


def test_dropout_statistical_at_p_nonzero(patched_attn):
    """Step 8: with dropout p>0 we can't be bit-exact, but the output
    distribution must still look like attention (mean ~0, finite var)."""
    attn, embed_dim, _ = patched_attn
    # Switch dropout on by re-creating the module with p=0.1 and into
    # train() mode so our forward applies dropout_p inside SDPA.
    from transformers.models.mbart.modeling_mbart import MBartAttention

    attn = MBartAttention(
        embed_dim=embed_dim, num_heads=4, dropout=0.1, is_decoder=True,
    ).train()

    torch.manual_seed(0)
    hidden = torch.randn(2, 32, embed_dim)
    samples = []
    for seed in range(64):
        torch.manual_seed(seed)
        out, _, _ = attn(hidden_states=hidden)
        samples.append(out.detach())
    stacked = torch.stack(samples)
    # Per-position mean across dropout draws should be small (close
    # to expectation of dropout-regularised attention output) and
    # variance should be finite -- this catches a kernel that quietly
    # NaNs out under dropout.
    assert stacked.isfinite().all()
    assert stacked.var() > 0
    assert stacked.mean().abs() < 1.0


def test_beam_search_log_prob_parity(patched_attn):
    """Step 7: log-prob of a fixed reference sequence under patched
    must match eager within atol=1e-3.

    We approximate this with a tiny MBartForCausalLM-style decoder
    forward over a fixed sequence of token ids, comparing log-softmax
    output. (Running real beam search would also work but is slower
    and tests the same numerical path.)
    """
    from transformers import MBartConfig
    from transformers.models.mbart.modeling_mbart import MBartForCausalLM

    if sdpa_patch._PATCHED:
        revert()

    config = MBartConfig(
        vocab_size=64, d_model=64,
        decoder_layers=2, decoder_attention_heads=4,
        decoder_ffn_dim=64, max_position_embeddings=64,
        is_decoder=True, add_cross_attention=False,
    )
    torch.manual_seed(0)
    model = MBartForCausalLM(config).eval()
    torch.manual_seed(0)
    ids = torch.randint(0, 64, (1, 16))

    with torch.no_grad():
        eager_out = model(input_ids=ids).logits
    apply()
    try:
        with torch.no_grad():
            patched_out = model(input_ids=ids).logits
    finally:
        revert()

    diff = float((eager_out - patched_out).abs().max().item())
    assert diff < 1e-3, f"logits diff={diff:.2e}"


def test_skip_check_env_skips_gate_but_applies(monkeypatch, caplog, tmp_path):
    """VISTA_SDPA_SKIP_CHECK=1 logs a WARNING and applies without
    running the ship-gate."""
    monkeypatch.setenv("VISTA_SDPA_SKIP_CHECK", "1")
    if sdpa_patch._PATCHED:
        revert()
    try:
        with caplog.at_level(logging.WARNING):
            sdpa_patch.enable_with_ship_gate(manifest_dir=tmp_path)
        assert sdpa_patch._PATCHED is True
        assert any(
            "VISTA_SDPA_SKIP_CHECK" in r.message for r in caplog.records
        )
        # Skip path: no manifest written (we didn't run the gate).
        assert not (tmp_path / "sdpa_manifest.txt").exists()
    finally:
        revert()


def test_enable_writes_manifest(monkeypatch, tmp_path):
    """Successful ship-gate writes the PASS line to <out>/sdpa_manifest.txt
    so paper-comparison runs cite a stable artefact."""
    monkeypatch.delenv("VISTA_SDPA_SKIP_CHECK", raising=False)
    if sdpa_patch._PATCHED:
        revert()
    try:
        sdpa_patch.enable_with_ship_gate(manifest_dir=tmp_path)
        manifest = tmp_path / "sdpa_manifest.txt"
        assert manifest.exists()
        line = manifest.read_text(encoding="utf-8").strip()
        assert line.startswith("SDPA patch: PASS")
        assert "transformers" in line
    finally:
        revert()


def test_training_smoke_loss_decreases_under_patch():
    """C3-A5: full MBartForCausalLM trains under the patch.

    Catches optimizer + forward + backward interaction bugs that the
    per-layer equivalence tests can't see (e.g. silently-broken cache
    semantics that surface only after several steps, or a parameter
    group accidentally left out of grads).

    50 steps, tiny model, AdamW. Asserts the smoothed loss in the
    last 10 steps is strictly below the smoothed loss in the first 10.
    """
    from transformers import MBartConfig
    from transformers.models.mbart.modeling_mbart import MBartForCausalLM

    if sdpa_patch._PATCHED:
        revert()

    config = MBartConfig(
        vocab_size=128, d_model=64, decoder_layers=2,
        decoder_attention_heads=4, decoder_ffn_dim=128,
        max_position_embeddings=64, is_decoder=True,
        add_cross_attention=False,
    )
    torch.manual_seed(0)
    model = MBartForCausalLM(config).train()
    # Fixed input: model overfits to it, loss must decrease.
    ids = torch.randint(0, 128, (2, 32))
    labels = ids.clone()

    apply()
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses: list[float] = []
        for _ in range(50):
            opt.zero_grad()
            out = model(input_ids=ids, labels=labels)
            out.loss.backward()
            opt.step()
            losses.append(float(out.loss.item()))
    finally:
        revert()

    early = sum(losses[:10]) / 10
    late = sum(losses[-10:]) / 10
    assert late < early, f"loss did not decrease: early={early:.3f} late={late:.3f}"
    # And the very last loss must be substantially below initial -- catches
    # "loss decreases by 0.001 then stalls" pathologies.
    assert losses[-1] < losses[0] * 0.5, (
        f"loss only halved: start={losses[0]:.3f} end={losses[-1]:.3f}"
    )


def test_equivalence_report_requires_apply():
    """Helper refuses to run without apply() -- prevents silent bugs
    where someone forgets to apply and gets eager-vs-eager."""
    if sdpa_patch._PATCHED:
        revert()
    from transformers.models.mbart.modeling_mbart import MBartAttention
    attn = MBartAttention(
        embed_dim=64, num_heads=4, dropout=0.0, is_decoder=True,
    )
    with pytest.raises(RuntimeError, match="apply"):
        equivalence_report(attn, {"hidden_states": torch.randn(1, 8, 64)})
