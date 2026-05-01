"""mBART decoder wrapper.

Wraps the decoder half of ``facebook/mbart-large-50`` (paper says "mBART-
based decoder"). At construction time:

1. Load the pretrained ``MBartForCausalLM`` config.
2. Override ``vocab_size`` to match our reduced EN-only vocab (subwords +
   spatial + special tokens).
3. Build a fresh ``MBartForCausalLM`` from that config so the embedding
   matrix has the right shape.
4. (Optional) ``load_pretrained_body=True`` copies non-embedding weights
   from the pretrained checkpoint, leaving embeddings randomly initialised
   for the new vocab.

For unit tests we use :func:`small_random_decoder` to instantiate a tiny
config without downloading any checkpoint.
"""
from __future__ import annotations

import logging

import torch
from torch import Tensor, nn
from transformers import MBartConfig, MBartForCausalLM

LOG = logging.getLogger(__name__)


def small_random_decoder(
    vocab_size: int,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    ffn_dim: int = 128,
    max_position_embeddings: int = 256,
    attn_implementation: str = "eager",
) -> MBartDecoder:
    """Build a tiny randomly-initialised :class:`MBartDecoder` for tests.
    Skips the ~610 MB pretrained download."""
    cfg = MBartConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        encoder_layers=1,           # unused -- we only run the decoder side
        decoder_layers=n_layers,
        decoder_attention_heads=n_heads,
        encoder_attention_heads=n_heads,
        decoder_ffn_dim=ffn_dim,
        encoder_ffn_dim=ffn_dim,
        max_position_embeddings=max_position_embeddings,
        is_decoder=True,
        is_encoder_decoder=False,
        add_cross_attention=True,
        tie_word_embeddings=True,
    )
    cfg._attn_implementation = attn_implementation
    model = MBartForCausalLM(cfg)
    return MBartDecoder.from_model(model, d_model=d_model, vocab_size=vocab_size)


class MBartDecoder(nn.Module):
    """Thin wrapper around :class:`transformers.MBartForCausalLM` so the
    rest of the codebase deals with a plain ``nn.Module`` interface.

    Inputs to :meth:`forward`:

    :param input_ids: ``(B, T)`` decoder input ids (already shifted).
    :param encoder_hidden_states: ``(B, S, d_model)`` from the encoder.
    :param attention_mask: optional decoder padding mask.
    :param encoder_attention_mask: optional encoder padding mask.

    Returns logits of shape ``(B, T, vocab_size)``.
    """

    def __init__(self, model: MBartForCausalLM, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.model = model
        self.d_model = d_model
        self.vocab_size = vocab_size
        n = sum(p.numel() for p in self.parameters())
        LOG.info("MBartDecoder initialised: %.2fM params, vocab=%d, d_model=%d",
                 n / 1e6, vocab_size, d_model)

    @classmethod
    def from_model(
        cls,
        model: MBartForCausalLM,
        *,
        d_model: int,
        vocab_size: int,
    ) -> MBartDecoder:
        return cls(model=model, d_model=d_model, vocab_size=vocab_size)

    @classmethod
    def from_pretrained_mbart50(
        cls,
        vocab_size: int,
        decoder_layers: int = 12,
        max_position_embeddings: int = 4096,
        load_pretrained_body: bool = True,
        attn_implementation: str = "eager",
    ) -> MBartDecoder:
        """Build the paper's decoder: 12-layer mBART-50 decoder with vocab
        resized to our EN-only vocab. If ``load_pretrained_body`` is True
        the transformer body weights are copied from
        ``facebook/mbart-large-50``; embeddings are always reinitialised
        for the new vocab."""
        ckpt = "facebook/mbart-large-50"
        ref = MBartForCausalLM.from_pretrained(ckpt) if load_pretrained_body else None

        cfg = MBartConfig.from_pretrained(ckpt)
        cfg.vocab_size = vocab_size
        cfg.decoder_layers = decoder_layers
        cfg.max_position_embeddings = max_position_embeddings
        cfg.is_decoder = True
        cfg.is_encoder_decoder = False
        cfg.add_cross_attention = True
        cfg.tie_word_embeddings = True
        cfg._attn_implementation = attn_implementation

        model = MBartForCausalLM(cfg)
        if ref is not None:
            _copy_body_weights(src=ref, dst=model)
        return cls(model=model, d_model=cfg.d_model, vocab_size=vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> Tensor:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=None,                          # loss is computed externally
            use_cache=False,
        )
        return out.logits

    @torch.no_grad()
    def generate_greedy(
        self,
        prompt_ids: Tensor,
        encoder_hidden_states: Tensor,
        eos_id: int,
        max_new_tokens: int = 512,
        encoder_attention_mask: Tensor | None = None,
        num_beams: int = 1,
        pad_id: int | None = None,
    ) -> Tensor:
        """KV-cache-backed generation via HuggingFace ``generate``.

        Replaces the earlier O(T^2) hand-rolled loop -- on real eval-length
        sequences (max_new_tokens=4096) this is a 10-50x speedup. Set
        ``num_beams > 1`` for beam search.

        ``pad_id`` MUST differ from ``eos_id``; otherwise HuggingFace's
        ``generate`` cannot distinguish padding from end-of-sequence and
        stops at step 1. Defaults to ``0`` if not provided.
        """
        if pad_id is None:
            pad_id = 0
        if pad_id == eos_id:
            raise ValueError(
                f"pad_id ({pad_id}) must differ from eos_id ({eos_id}); "
                "passing the same id makes HF generate() stop immediately."
            )
        prompt_len = prompt_ids.shape[1]
        attention_mask = torch.ones_like(prompt_ids)
        out = self.model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            use_cache=True,
        )
        return out[:, prompt_len:]


def _copy_body_weights(src: MBartForCausalLM, dst: MBartForCausalLM) -> None:
    """Copy every parameter from ``src`` whose shape matches the
    corresponding parameter in ``dst``. Embedding-shaped tensors that don't
    match (different vocab) are left at their fresh random init."""
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    copied = 0
    skipped: list[str] = []
    for k, v in src_state.items():
        if k in dst_state and dst_state[k].shape == v.shape:
            dst_state[k] = v
            copied += 1
        else:
            skipped.append(k)
    dst.load_state_dict(dst_state, strict=False)
    LOG.info("Copied %d/%d weights from pretrained mBART; %d skipped (shape mismatch)",
             copied, len(src_state), len(skipped))


__all__ = ["MBartDecoder", "small_random_decoder"]
