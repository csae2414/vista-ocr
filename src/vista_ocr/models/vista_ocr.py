"""End-to-end VISTA-OCR module wiring encoder + decoder.

Encoder output is ``(B, S, 1024)`` flattened tokens; decoder consumes them
as ``encoder_hidden_states`` for cross-attention. d_model on both sides is
1024, so no projection is needed (matches the paper).
"""
from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from vista_ocr.models.decoder import MBartDecoder
from vista_ocr.models.encoder import FCNEncoderWidther

LOG = logging.getLogger(__name__)


class VistaOCR(nn.Module):
    """Combined encoder + decoder.

    Both halves are plain ``nn.Module`` instances so the training loop
    can apply gradient checkpointing or freeze either side independently.
    """

    def __init__(self, encoder: FCNEncoderWidther, decoder: MBartDecoder) -> None:
        super().__init__()
        if decoder.d_model != encoder.OUT_CHANNELS:
            raise ValueError(
                f"Encoder out_channels ({encoder.OUT_CHANNELS}) must equal "
                f"decoder d_model ({decoder.d_model}) for direct cross-attention"
            )
        self.encoder = encoder
        self.decoder = decoder
        n = sum(p.numel() for p in self.parameters())
        LOG.info("VistaOCR total params: %.2fM", n / 1e6)

    def encode(self, images: Tensor) -> Tensor:
        return self.encoder(images)               # (B, S, d_model)

    def forward(
        self,
        images: Tensor,
        decoder_input_ids: Tensor,
        decoder_attention_mask: Tensor | None = None,
    ) -> Tensor:
        memory = self.encode(images)
        return self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=memory,
        )

    def freeze_decoder(self, freeze: bool = True) -> None:
        """Stage-1a calibration: freeze the decoder while encoder catches up."""
        for p in self.decoder.parameters():
            p.requires_grad = not freeze

    @torch.no_grad()
    def generate(
        self,
        images: Tensor,
        prompt_ids: Tensor,
        eos_id: int,
        max_new_tokens: int = 512,
    ) -> Tensor:
        memory = self.encode(images)
        return self.decoder.generate_greedy(
            prompt_ids=prompt_ids,
            encoder_hidden_states=memory,
            eos_id=eos_id,
            max_new_tokens=max_new_tokens,
        )


__all__ = ["VistaOCR"]
