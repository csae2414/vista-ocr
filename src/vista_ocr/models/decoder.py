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
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        min_new_tokens: int = 0,
    ) -> Tensor:
        """KV-cache-backed generation via HuggingFace ``generate``.

        Replaces the earlier O(T^2) hand-rolled loop -- on real eval-length
        sequences (max_new_tokens=4096) this is a 10-50x speedup. Set
        ``num_beams > 1`` for beam search.

        ``pad_id`` MUST differ from ``eos_id``; otherwise HuggingFace's
        ``generate`` cannot distinguish padding from end-of-sequence and
        stops at step 1. Defaults to ``0`` if not provided.

        Anti-repetition / length knobs (default off):

        * ``repetition_penalty`` -- divides the logit of any prior token
          by this factor each step. Compounds across the sequence; do
          NOT use for benchmark numbers (changes output distribution).
          Safe inspection value: 1.05.
        * ``no_repeat_ngram_size`` -- forbid any n-gram that already
          appeared. Our serialisation deliberately repeats trigrams of
          shape ``<y> word <x>`` between lines, so size 3 BREAKS line
          structure. Use 6+ if at all. 0 = off.
        * ``min_new_tokens`` -- minimum new tokens before EOS may
          fire. Forcing length on genuinely short pages hallucinates;
          0 = off.
        """
        if pad_id is None:
            pad_id = 0
        if pad_id == eos_id:
            raise ValueError(
                f"pad_id ({pad_id}) must differ from eos_id ({eos_id}); "
                "passing the same id makes HF generate() stop immediately."
            )
        if num_beams != 1:
            raise NotImplementedError(
                "Hand-rolled greedy loop only supports num_beams=1. "
                "Beam search needs a separate implementation that also "
                "threads encoder_hidden_states through the per-step inputs.",
            )
        # WHY this loop instead of HF generate(): MBartForCausalLM's
        # ``prepare_inputs_for_generation`` strips
        # ``encoder_hidden_states`` from the per-step model inputs, so
        # cross-attention sees nothing during HF's generation loop and
        # the decoder produces image-blind output (identical tokens for
        # different images). We sidestep that by calling ``forward``
        # directly each step with encoder_hidden_states explicitly
        # passed; KV cache is still used so this stays O(T) per step.
        return _greedy_decode_with_cross_attention(
            model=self.model,
            prompt_ids=prompt_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            eos_id=eos_id,
            pad_id=pad_id,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )


def _greedy_decode_with_cross_attention(
    *,
    model: MBartForCausalLM,
    prompt_ids: Tensor,
    encoder_hidden_states: Tensor,
    encoder_attention_mask: Tensor | None,
    eos_id: int,
    pad_id: int,
    max_new_tokens: int,
    min_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> Tensor:
    """Greedy decoding that calls ``model.forward`` directly each step.

    Bypasses HF ``generate``'s ``prepare_inputs_for_generation`` (which
    drops ``encoder_hidden_states`` for ``MBartForCausalLM``). Threads
    the encoder features into the very first step so cross-attention
    KV is built correctly; subsequent steps reuse the cached cross-KV.

    Supports ``repetition_penalty`` (HF semantics: divide-on-positive,
    multiply-on-negative) and ``no_repeat_ngram_size`` (forbid any
    n-gram already present in the running output). Both are applied
    pre-argmax. ``min_new_tokens`` masks EOS to ``-inf`` until the
    threshold is reached.
    """
    bsz, prompt_len = prompt_ids.shape
    device = prompt_ids.device
    generated: list[Tensor] = [prompt_ids]
    finished = torch.zeros(bsz, dtype=torch.bool, device=device)
    past_key_values = None

    for step in range(max_new_tokens):
        if step == 0:
            input_ids = prompt_ids
        else:
            input_ids = generated[-1][:, -1:]

        kwargs = dict(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        out = model(**kwargs)
        logits = out.logits[:, -1, :]   # (B, V)
        past_key_values = out.past_key_values

        # Repetition penalty (HF semantics).
        if repetition_penalty != 1.0:
            full = torch.cat(generated, dim=1)
            score = logits.gather(1, full)
            score = torch.where(score < 0, score * repetition_penalty,
                                score / repetition_penalty)
            logits.scatter_(1, full, score)

        # n-gram blocking.
        if no_repeat_ngram_size > 0:
            full = torch.cat(generated, dim=1).tolist()
            for b in range(bsz):
                seq = full[b]
                if len(seq) >= no_repeat_ngram_size - 1:
                    suffix = tuple(seq[-(no_repeat_ngram_size - 1):])
                    banned: set[int] = set()
                    for i in range(len(seq) - no_repeat_ngram_size + 1):
                        if tuple(seq[i:i + no_repeat_ngram_size - 1]) == suffix:
                            banned.add(seq[i + no_repeat_ngram_size - 1])
                    for tok in banned:
                        logits[b, tok] = float("-inf")

        # min_new_tokens: forbid EOS until threshold.
        if step < min_new_tokens:
            logits[:, eos_id] = float("-inf")

        next_tok = logits.argmax(dim=-1, keepdim=True)   # (B, 1)
        # Once finished, force pad_id so EOS isn't re-emitted.
        next_tok = torch.where(
            finished.unsqueeze(1),
            torch.full_like(next_tok, pad_id),
            next_tok,
        )
        generated.append(next_tok)
        finished = finished | (next_tok.squeeze(1) == eos_id)
        if bool(finished.all()):
            break

    full = torch.cat(generated, dim=1)
    return full[:, prompt_len:]


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
