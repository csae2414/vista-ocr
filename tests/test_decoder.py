"""Decoder + end-to-end VistaOCR tests using a tiny random mBART config."""
from __future__ import annotations

import pytest
import torch

from vista_ocr.models.decoder import MBartDecoder, small_random_decoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR


@pytest.fixture(scope="module")
def tiny_decoder() -> MBartDecoder:
    torch.manual_seed(0)
    return small_random_decoder(vocab_size=200, d_model=64, n_layers=2, n_heads=4)


def test_decoder_forward_shape(tiny_decoder: MBartDecoder):
    batch, tgt_len, src_len = 2, 7, 5
    input_ids = torch.randint(0, tiny_decoder.vocab_size, (batch, tgt_len))
    memory = torch.randn(batch, src_len, tiny_decoder.d_model)
    logits = tiny_decoder(input_ids=input_ids, encoder_hidden_states=memory)
    assert logits.shape == (batch, tgt_len, tiny_decoder.vocab_size)


def test_decoder_backward(tiny_decoder: MBartDecoder):
    input_ids = torch.randint(0, tiny_decoder.vocab_size, (1, 6))
    memory = torch.randn(1, 4, tiny_decoder.d_model, requires_grad=True)
    logits = tiny_decoder(input_ids=input_ids, encoder_hidden_states=memory)
    logits.sum().backward()
    assert memory.grad is not None
    assert torch.isfinite(memory.grad).all()


def test_decoder_greedy_generation_terminates(tiny_decoder: MBartDecoder):
    prompt = torch.randint(0, tiny_decoder.vocab_size, (1, 3))
    memory = torch.randn(1, 5, tiny_decoder.d_model)
    out = tiny_decoder.generate_greedy(
        prompt_ids=prompt,
        encoder_hidden_states=memory,
        eos_id=0,
        pad_id=1,                 # must differ from eos_id
        max_new_tokens=8,
    )
    assert out.shape[0] == 1
    assert 0 < out.shape[1] <= 8


def test_vista_ocr_end_to_end():
    """Mini VISTA-OCR with d_model=1024 on a tiny image. Validates that
    encoder→decoder shapes line up at the paper's d_model."""
    torch.manual_seed(0)
    enc = FCNEncoderWidther(input_channels=1, dropout=0.0)
    # Decoder must use the paper's d_model (1024) to share cross-attention.
    dec = small_random_decoder(vocab_size=128, d_model=1024, n_layers=1, n_heads=4)
    model = VistaOCR(encoder=enc, decoder=dec)

    images = torch.randn(1, 1, 64, 64)
    decoder_input_ids = torch.randint(0, 128, (1, 5))
    logits = model(images, decoder_input_ids)
    assert logits.shape == (1, 5, 128)


def test_generate_rejects_pad_eq_eos(tiny_decoder: MBartDecoder):
    """HuggingFace generate() needs pad != eos to avoid stopping at step 1."""
    prompt = torch.randint(0, tiny_decoder.vocab_size, (1, 2))
    memory = torch.randn(1, 4, tiny_decoder.d_model)
    with pytest.raises(ValueError, match="pad_id"):
        tiny_decoder.generate_greedy(
            prompt_ids=prompt, encoder_hidden_states=memory,
            eos_id=3, pad_id=3, max_new_tokens=4,
        )


def test_freeze_decoder_flag():
    enc = FCNEncoderWidther(input_channels=1, dropout=0.0)
    dec = small_random_decoder(vocab_size=64, d_model=1024, n_layers=1, n_heads=4)
    model = VistaOCR(enc, dec)
    model.freeze_decoder(True)
    assert all(not p.requires_grad for p in model.decoder.parameters())
    assert any(p.requires_grad for p in model.encoder.parameters())
    model.freeze_decoder(False)
    assert all(p.requires_grad for p in model.decoder.parameters())
