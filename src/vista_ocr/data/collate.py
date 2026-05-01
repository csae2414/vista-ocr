"""Batch collation.

Builds decoder input/label tensors from a list of
:class:`vista_ocr.data.types.Sample` objects. Handles task-conditional
prompt prepending and pads to the longest sequence in the batch."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from PIL import Image

from vista_ocr.data.preprocess import (
    PreprocessConfig,
    pad_to_multiple,
    resize_to_canvas,
    to_tensor,
)
from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.tokenizer import VistaTokenizer

LOG = logging.getLogger(__name__)


@dataclass
class Batch:
    images: torch.Tensor              # (B, 1, H, W)
    decoder_input_ids: torch.Tensor   # (B, T)
    labels: torch.Tensor              # (B, T) — pad positions = pad_id
    prompt_mask: torch.Tensor         # (B, T) — True where token is prompt
    pad_id: int


# Hard cap on the decoder sequence length we feed into training. Pages
# with hundreds of lines can produce >4000 tokens, and a particular bf16
# self-attention path in transformers 4.44 raises CUBLAS_STATUS_EXECUTION_FAILED
# at those lengths on a 24GB RTX 3090. Truncating preserves a meaningful
# learning signal -- the prompt + first portion of lines fit fine.
MAX_TARGET_TOKENS: int = 2048


def build_target_ids(tokenizer: VistaTokenizer, sample: Sample) -> tuple[list[int], int]:
    """Construct ``[prompt..., output..., </s>]`` and return the prompt
    length so the trainer can mask it out of the loss."""
    if sample.task == "ocr":
        prompt = tokenizer.build_ocr_prompt(with_layout=False)
        output = []
        for line in sample.lines:
            output.extend(tokenizer.encode_text(line.text))
    elif sample.task == "ocr_layout":
        prompt = tokenizer.build_ocr_prompt(with_layout=True)
        output = tokenizer.serialize_lines(sample.lines, scheme="original")
    elif sample.task == "region_ocr":
        if sample.query_bbox is None:
            raise ValueError("region_ocr task requires query_bbox")
        prompt = tokenizer.build_region_ocr_prompt(sample.query_bbox)
        # Output: just the text inside the queried bbox (lines that fall in it).
        output = []
        for line in sample.lines:
            if _bbox_contains(sample.query_bbox, line.bbox):
                output.extend(tokenizer.encode_text(line.text))
    elif sample.task == "find_it":
        if sample.query_text is None:
            raise ValueError("find_it task requires query_text")
        prompt = tokenizer.build_find_it_prompt(sample.query_text)
        output = []
        for line in sample.lines:
            if sample.query_text.strip() in line.text:
                x1, y1, x2, y2 = line.bbox
                output.append(tokenizer.piece_to_id(tokenizer.grid.x_token(x1)))
                output.append(tokenizer.piece_to_id(tokenizer.grid.y_token(y1)))
                output.append(tokenizer.piece_to_id(tokenizer.grid.x_token(x2)))
                output.append(tokenizer.piece_to_id(tokenizer.grid.y_token(y2)))
    else:
        raise ValueError(f"Unknown task: {sample.task}")

    seq = [tokenizer.bos_id, *prompt, *output, tokenizer.eos_id]
    prompt_len = 1 + len(prompt)  # bos + prompt are non-supervised positions

    # Truncate runaway sequences. Keep the prompt intact (otherwise the
    # task-conditioning header is lost) and the trailing eos_id; clip the
    # middle.
    if len(seq) > MAX_TARGET_TOKENS:
        keep_after_prompt = MAX_TARGET_TOKENS - prompt_len - 1
        seq = seq[:prompt_len] + seq[prompt_len: prompt_len + keep_after_prompt] + [tokenizer.eos_id]
    return seq, prompt_len


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2


def collate(
    samples: list[Sample],
    tokenizer: VistaTokenizer,
    pre_cfg: PreprocessConfig,
) -> Batch:
    """Image preprocessing + token batching. Pads images to the largest in
    the batch (multiple of ``pre_cfg.pad_multiple``)."""
    image_tensors: list[torch.Tensor] = []
    seqs: list[list[int]] = []
    prompt_lens: list[int] = []

    for s in samples:
        if isinstance(s.image, torch.Tensor):
            t = s.image if s.image.dim() == 4 else s.image.unsqueeze(0)
        else:
            img, _, _ = resize_to_canvas(s.image, pre_cfg)
            img, _ = pad_to_multiple(img, pre_cfg.pad_multiple)
            t = to_tensor(img)
        image_tensors.append(t)
        seq, plen = build_target_ids(tokenizer, s)
        seqs.append(seq)
        prompt_lens.append(plen)

    max_h = max(t.shape[-2] for t in image_tensors)
    max_w = max(t.shape[-1] for t in image_tensors)
    images = torch.ones(len(samples), 1, max_h, max_w)
    for i, t in enumerate(image_tensors):
        h, w = t.shape[-2], t.shape[-1]
        images[i, :, :h, :w] = t

    max_t = max(len(s) for s in seqs)
    pad_id = tokenizer.pad_id
    input_ids = torch.full((len(samples), max_t), pad_id, dtype=torch.long)
    labels = torch.full((len(samples), max_t), pad_id, dtype=torch.long)
    prompt_mask = torch.zeros_like(labels, dtype=torch.bool)
    for i, (seq, plen) in enumerate(zip(seqs, prompt_lens)):
        # Standard teacher forcing: input = seq[:-1], label = seq[1:].
        for j in range(len(seq) - 1):
            input_ids[i, j] = seq[j]
            labels[i, j] = seq[j + 1]
            if j < plen - 1:
                prompt_mask[i, j] = True

    return Batch(
        images=images,
        decoder_input_ids=input_ids,
        labels=labels,
        prompt_mask=prompt_mask,
        pad_id=pad_id,
    )
