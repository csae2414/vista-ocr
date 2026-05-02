"""Initialise the VistaOCR text decoder from Donut's BART text decoder.

VISTA-OCR's paper (arXiv:2504.03621) Section 3.2 says the decoder is
initialised "using those of Donut". This module wires that up: it
loads the BART text decoder from ``naver-clova-ix/donut-base`` (4
layers, ``d_model=1024``, matching our decoder geometry exactly),
strips the parts we cannot transfer (vocab-shaped embedding + lm_head;
Donut's vocab is 57 525 vs our spatial-token-augmented 16 000), and
copies the remaining transformer body into our :class:`MBartDecoder`.

Why a separate module
---------------------

The plain :func:`vista_ocr.models.decoder._copy_body_weights` already
does name-and-shape matching, but does not:

* surface a per-category copy/skip count (``embed_tokens``, ``lm_head``,
  ``cross_attn``, etc.) so an operator can verify the body actually
  transferred,
* refuse to fall back to the random init when *zero* parameters
  transfer (silent failure mode),
* offer a fixture-based load so unit tests run without the 700 MB
  download.

This module wraps those concerns and exposes one public function:
:func:`init_decoder_from_donut`.

Why we don't transfer the embedding / lm_head
---------------------------------------------

Our text vocabulary is 16 000 SentencePiece pieces plus spatial
tokens. Donut's is 57 525. The token IDs are not aligned, so
re-using the embedding rows would map random glyphs to random
words and harm training. We *intentionally* skip them; the
randomly-initialised embedding/head are re-trained from scratch
in stage 1 (the encoder calibration stage). The transformer body
-- self-attention, cross-attention, FFN, layer norms -- is what we
actually want for the language prior.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch

LOG = logging.getLogger(__name__)

# State-dict keys we expect to drop because they encode the source
# model's vocabulary (different from ours).
_VOCAB_SHAPED_KEYS: tuple[str, ...] = (
    "model.decoder.embed_tokens.weight",
    "lm_head.weight",
    "lm_head.bias",
    "final_logits_bias",
)


def _remap_donut_to_mbart_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map Donut/BART decoder keys to our :class:`MBartDecoder` shape.

    For ``naver-clova-ix/donut-base`` the BART decoder already uses
    the ``model.decoder.X`` prefix that ``MBartForCausalLM`` also
    uses, so the mapping is largely identity. The function still
    exists because:

    * It strips vocab-shaped keys explicitly (see
      :data:`_VOCAB_SHAPED_KEYS`); doing this here keeps
      :func:`init_decoder_from_donut` declarative.
    * It is the single seam at which we can adapt to future
      transformers releases that rename modules.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k in _VOCAB_SHAPED_KEYS:
            continue
        # Donut decoder keys are already 'model.decoder.X' shaped.
        # Future-proof: strip any leading 'decoder.' prefix and
        # re-prefix with 'model.decoder.' so a raw BartDecoder
        # state_dict (without the BartForCausalLM wrapper) also works.
        if k.startswith("model.decoder."):
            out[k] = v
        elif k.startswith("decoder."):
            out["model." + k] = v
        # Anything else (encoder.*, etc.) is dropped silently.
    return out


def _copy_state_dict_shape_matched(
    src: Mapping[str, torch.Tensor],
    dst_module: torch.nn.Module,
) -> dict[str, Any]:
    """Copy ``src`` tensors into ``dst_module`` where keys exist and
    shapes match.

    Returns a report dict with ``copied``, ``skipped_shape``,
    ``skipped_missing``, and the categorised tensor names. Used by
    callers that need to verify a meaningful transfer happened.
    """
    dst_state = dst_module.state_dict()
    copied: list[str] = []
    skipped_shape: list[str] = []
    skipped_missing: list[str] = []
    for k, v in src.items():
        if k not in dst_state:
            skipped_missing.append(k)
            continue
        if dst_state[k].shape != v.shape:
            skipped_shape.append(k)
            continue
        dst_state[k] = v
        copied.append(k)
    dst_module.load_state_dict(dst_state, strict=False)
    return {
        "copied": copied,
        "skipped_shape": skipped_shape,
        "skipped_missing": skipped_missing,
    }


def init_decoder_from_donut(
    decoder: torch.nn.Module,
    *,
    donut_state_dict: Mapping[str, torch.Tensor] | None = None,
    repo: str = "naver-clova-ix/donut-base",
    cache_dir: str | None = None,
    require_min_overlap: int = 50,
) -> dict[str, Any]:
    """Initialise ``decoder.model`` (an ``MBartForCausalLM``) from
    Donut's BART decoder weights.

    :param decoder: a :class:`vista_ocr.models.decoder.MBartDecoder`
        whose ``.model`` is an ``MBartForCausalLM``.
    :param donut_state_dict: optional pre-loaded state dict, primarily
        for tests. When ``None`` we ``from_pretrained`` the ``repo``.
    :param repo: HF repo of the source weights. Default
        ``naver-clova-ix/donut-base`` (the paper's choice).
    :param cache_dir: HF cache directory passed through to
        ``from_pretrained`` for offline/known-path loads.
    :param require_min_overlap: refuse to silently fall back to the
        random init if fewer than this many tensors copied. Catches
        breakage from a future module rename rather than failing
        silently mid-training. Default 50 is comfortably below the
        ~100 we actually expect.

    :returns: report dict from
        :func:`_copy_state_dict_shape_matched`, augmented with
        ``n_copied`` for convenience.
    """
    if donut_state_dict is None:
        try:
            from transformers import VisionEncoderDecoderModel  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers is required to load Donut weights",
            ) from exc
        LOG.info("Loading Donut decoder from %s (this may take ~700 MB on first run)", repo)
        full_model = VisionEncoderDecoderModel.from_pretrained(
            repo, cache_dir=cache_dir,
        )
        donut_state_dict = full_model.decoder.state_dict()
        # Encoder is large (Swin); drop it eagerly so it isn't held
        # for the rest of training.
        del full_model

    remapped = _remap_donut_to_mbart_keys(donut_state_dict)
    report = _copy_state_dict_shape_matched(remapped, decoder.model)
    n_copied = len(report["copied"])
    report["n_copied"] = n_copied
    if n_copied < require_min_overlap:
        raise RuntimeError(
            f"Donut init copied only {n_copied} tensors (< "
            f"{require_min_overlap} required). The decoder is therefore "
            f"effectively still random. transformers naming may have "
            f"changed; inspect skipped_missing={report['skipped_missing'][:5]}."
        )
    LOG.info(
        "Donut decoder init: copied %d tensors, skipped %d (shape mismatch), "
        "skipped %d (missing in dst). Vocab-shaped tensors intentionally "
        "dropped; embeddings/lm-head are re-trained from random init.",
        n_copied,
        len(report["skipped_shape"]),
        len(report["skipped_missing"]),
    )
    return report


__all__ = [
    "init_decoder_from_donut",
    "_remap_donut_to_mbart_keys",
    "_copy_state_dict_shape_matched",
]
