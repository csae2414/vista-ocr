"""Tests for :mod:`vista_ocr.models.donut_init`.

Pure-Python unit tests + a fixture-based integration test that
exercises ``init_decoder_from_donut`` on a synthesized state-dict so
CI never has to download the 700 MB Donut weights.
"""
from __future__ import annotations

import logging

import pytest
import torch

from vista_ocr.models.decoder import small_random_decoder
from vista_ocr.models.donut_init import (
    _VOCAB_SHAPED_KEYS,
    _copy_state_dict_shape_matched,
    _remap_donut_to_mbart_keys,
    init_decoder_from_donut,
)


# ---------------------------------------------------------------------
# _remap_donut_to_mbart_keys
# ---------------------------------------------------------------------

class TestRemap:
    def test_canonical_mbart_prefix_passes_through(self):
        sd = {
            "model.decoder.layers.0.self_attn.q_proj.weight": torch.zeros(2),
            "model.decoder.embed_positions.weight": torch.zeros(2),
        }
        out = _remap_donut_to_mbart_keys(sd)
        assert out == sd

    def test_bare_decoder_prefix_gets_model_dot(self):
        sd = {"decoder.layers.0.self_attn.q_proj.weight": torch.zeros(2)}
        out = _remap_donut_to_mbart_keys(sd)
        assert "model.decoder.layers.0.self_attn.q_proj.weight" in out
        assert "decoder.layers.0.self_attn.q_proj.weight" not in out

    def test_vocab_shaped_keys_stripped(self):
        sd = {
            "model.decoder.embed_tokens.weight": torch.zeros(57525, 1024),
            "model.decoder.layers.0.self_attn.q_proj.weight": torch.zeros(2),
            "lm_head.weight": torch.zeros(57525, 1024),
            "final_logits_bias": torch.zeros(57525),
        }
        out = _remap_donut_to_mbart_keys(sd)
        for k in _VOCAB_SHAPED_KEYS:
            assert k not in out
        assert "model.decoder.layers.0.self_attn.q_proj.weight" in out

    def test_encoder_keys_dropped(self):
        sd = {
            "encoder.layers.0.self_attn.q_proj.weight": torch.zeros(2),
            "model.decoder.layers.0.self_attn.q_proj.weight": torch.zeros(2),
        }
        out = _remap_donut_to_mbart_keys(sd)
        assert "encoder.layers.0.self_attn.q_proj.weight" not in out
        assert "model.decoder.layers.0.self_attn.q_proj.weight" in out

    def test_idempotent(self):
        sd = {"model.decoder.layers.0.fc1.weight": torch.ones(8, 8)}
        once = _remap_donut_to_mbart_keys(sd)
        twice = _remap_donut_to_mbart_keys(once)
        assert once.keys() == twice.keys()
        assert torch.equal(once["model.decoder.layers.0.fc1.weight"],
                           twice["model.decoder.layers.0.fc1.weight"])


# ---------------------------------------------------------------------
# _copy_state_dict_shape_matched
# ---------------------------------------------------------------------

class TestCopyShapeMatched:
    def test_copies_matching_shape_and_name(self):
        # Build a tiny dst module: one parameter we can target.
        class _Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 2)

        dst = _Tiny()
        # Source has same key, same shape -> should copy.
        src = {"fc.weight": torch.ones(2, 4), "fc.bias": torch.ones(2)}
        report = _copy_state_dict_shape_matched(src, dst)
        assert "fc.weight" in report["copied"]
        assert "fc.bias" in report["copied"]
        assert torch.equal(dst.fc.weight, torch.ones(2, 4))
        assert torch.equal(dst.fc.bias, torch.ones(2))

    def test_shape_mismatch_is_skipped_not_raise(self):
        class _Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 2)

        dst = _Tiny()
        src = {"fc.weight": torch.ones(8, 4)}     # wrong shape
        report = _copy_state_dict_shape_matched(src, dst)
        assert "fc.weight" in report["skipped_shape"]
        assert "fc.weight" not in report["copied"]

    def test_missing_in_dst_is_categorised(self):
        class _Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 2)

        dst = _Tiny()
        src = {"not_in_dst.weight": torch.ones(4)}
        report = _copy_state_dict_shape_matched(src, dst)
        assert "not_in_dst.weight" in report["skipped_missing"]


# ---------------------------------------------------------------------
# init_decoder_from_donut (integration with synthetic state-dict)
# ---------------------------------------------------------------------

@pytest.fixture
def tiny_decoder():
    """A 2-layer, d_model=64 MBartDecoder for fast tests."""
    return small_random_decoder(
        vocab_size=128, d_model=64, n_layers=2, n_heads=4,
        ffn_dim=128, max_position_embeddings=64,
    )


def _build_synthetic_donut_state_dict(decoder) -> dict[str, torch.Tensor]:
    """Take the dst decoder's own state dict, fill every tensor with
    a deterministic non-zero value, and return it. Stand-in for a real
    Donut state dict in tests -- guarantees shape compatibility.

    Adds vocab-shaped tensors with the WRONG vocab dim so we can
    verify they are skipped by the remap.
    """
    sd = {k: torch.full_like(v, 0.5) for k, v in decoder.model.state_dict().items()}
    sd["model.decoder.embed_tokens.weight"] = torch.zeros(57525, decoder.d_model)
    sd["lm_head.weight"] = torch.zeros(57525, decoder.d_model)
    return sd


def test_init_from_donut_copies_body_tensors(tiny_decoder):
    pre = {k: v.clone() for k, v in tiny_decoder.model.state_dict().items()}
    src = _build_synthetic_donut_state_dict(tiny_decoder)
    report = init_decoder_from_donut(
        tiny_decoder, donut_state_dict=src, require_min_overlap=10,
    )
    assert report["n_copied"] >= 10
    # Body tensors got the 0.5 fill we put in src.
    post = tiny_decoder.model.state_dict()
    body_key = "model.decoder.layers.0.self_attn.q_proj.weight"
    assert torch.allclose(post[body_key], torch.full_like(post[body_key], 0.5))
    # And we changed something vs the random init.
    assert not torch.equal(pre[body_key], post[body_key])


def test_init_from_donut_skips_vocab_shaped_tensors(tiny_decoder):
    """Even though the synthetic src has 57525-shaped embed_tokens,
    the dst's small-vocab embed_tokens must NOT be overwritten."""
    pre_embed = tiny_decoder.model.state_dict()[
        "model.decoder.embed_tokens.weight"
    ].clone()
    src = _build_synthetic_donut_state_dict(tiny_decoder)
    init_decoder_from_donut(
        tiny_decoder, donut_state_dict=src, require_min_overlap=10,
    )
    post_embed = tiny_decoder.model.state_dict()[
        "model.decoder.embed_tokens.weight"
    ]
    assert torch.equal(pre_embed, post_embed)


def test_init_from_donut_idempotent(tiny_decoder):
    src = _build_synthetic_donut_state_dict(tiny_decoder)
    rep1 = init_decoder_from_donut(
        tiny_decoder, donut_state_dict=src, require_min_overlap=10,
    )
    rep2 = init_decoder_from_donut(
        tiny_decoder, donut_state_dict=src, require_min_overlap=10,
    )
    assert rep1["n_copied"] == rep2["n_copied"]


def test_init_from_donut_raises_when_overlap_too_small(tiny_decoder):
    # Empty source -> nothing copies -> RuntimeError.
    with pytest.raises(RuntimeError, match="copied only"):
        init_decoder_from_donut(
            tiny_decoder, donut_state_dict={}, require_min_overlap=1,
        )


def test_init_from_donut_logs_categorised_summary(tiny_decoder, caplog):
    src = _build_synthetic_donut_state_dict(tiny_decoder)
    with caplog.at_level(logging.INFO):
        init_decoder_from_donut(
            tiny_decoder, donut_state_dict=src, require_min_overlap=10,
        )
    msg = " ".join(r.message for r in caplog.records)
    assert "copied" in msg.lower()
    assert "vocab" in msg.lower()
