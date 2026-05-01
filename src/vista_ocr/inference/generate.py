"""Inference helpers.

Wraps :class:`VistaOCR.generate` with task-specific prompt builders and
output parsers. Greedy decoding only for now (PLAN §12.6 default); a beam
variant goes through ``model.decoder.model.generate`` once we need it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from vista_ocr.data.preprocess import (
    PreprocessConfig,
    pad_to_multiple,
    resize_to_canvas,
    to_tensor,
)
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.tokenizer import Line, VistaTokenizer

LOG = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    max_new_tokens: int = 4096
    pad_multiple: int = 32
    target_h: int = 3508
    target_w: int = 2480
    device: str = "cpu"
    # Anti-repetition / length knobs. All default off so benchmark
    # decoding stays comparable to the paper's. Only the inspection
    # script enables them with conservative values; do NOT turn them
    # on inside finetune_eval.py.
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    min_new_tokens: int = 0


def _prepare_image(img, cfg: InferenceConfig) -> torch.Tensor:
    pre_cfg = PreprocessConfig(
        target_h=cfg.target_h, target_w=cfg.target_w, pad_multiple=cfg.pad_multiple
    )
    img, _, _ = resize_to_canvas(img, pre_cfg)
    img, _ = pad_to_multiple(img, pre_cfg.pad_multiple)
    return to_tensor(img).to(cfg.device)


def _generate(
    model: VistaOCR,
    image_tensor: torch.Tensor,
    prompt_ids: list[int],
    tokenizer: VistaTokenizer,
    cfg: InferenceConfig,
) -> list[int]:
    prompt = torch.tensor([[tokenizer.bos_id, *prompt_ids]], dtype=torch.long, device=cfg.device)
    out = model.generate(
        images=image_tensor,
        prompt_ids=prompt,
        eos_id=tokenizer.eos_id,
        max_new_tokens=cfg.max_new_tokens,
        pad_id=tokenizer.pad_id,
        repetition_penalty=cfg.repetition_penalty,
        no_repeat_ngram_size=cfg.no_repeat_ngram_size,
        min_new_tokens=cfg.min_new_tokens,
    )
    return out[0].tolist()


def ocr_with_layout(
    model: VistaOCR,
    image,
    tokenizer: VistaTokenizer,
    cfg: InferenceConfig | None = None,
) -> list[Line]:
    cfg = cfg or InferenceConfig()
    image_tensor = _prepare_image(image, cfg)
    prompt = tokenizer.build_ocr_prompt(with_layout=True)
    ids = _generate(model, image_tensor, prompt, tokenizer, cfg)
    ids_clean = [i for i in ids if i != tokenizer.eos_id]
    return tokenizer.parse_original_output(ids_clean)


def region_ocr(
    model: VistaOCR,
    image,
    bbox: tuple[int, int, int, int],
    tokenizer: VistaTokenizer,
    cfg: InferenceConfig | None = None,
) -> str:
    cfg = cfg or InferenceConfig()
    image_tensor = _prepare_image(image, cfg)
    prompt = tokenizer.build_region_ocr_prompt(bbox)
    ids = _generate(model, image_tensor, prompt, tokenizer, cfg)
    ids_clean = [
        i for i in ids
        if i != tokenizer.eos_id and not tokenizer.is_spatial_id(i)
        and not tokenizer.is_special_id(i)
    ]
    return tokenizer.decode_ids(ids_clean)


def find_it(
    model: VistaOCR,
    image,
    query: str,
    tokenizer: VistaTokenizer,
    cfg: InferenceConfig | None = None,
) -> list[tuple[int, int, int, int]]:
    cfg = cfg or InferenceConfig()
    image_tensor = _prepare_image(image, cfg)
    prompt = tokenizer.build_find_it_prompt(query)
    ids = _generate(model, image_tensor, prompt, tokenizer, cfg)
    boxes: list[tuple[int, int, int, int]] = []
    spatial = [i for i in ids if tokenizer.is_spatial_id(i)]
    for k in range(0, len(spatial) - 3, 4):
        x1 = tokenizer._token_to_px(tokenizer.id_to_piece(spatial[k]), axis="x")
        y1 = tokenizer._token_to_px(tokenizer.id_to_piece(spatial[k + 1]), axis="y")
        x2 = tokenizer._token_to_px(tokenizer.id_to_piece(spatial[k + 2]), axis="x")
        y2 = tokenizer._token_to_px(tokenizer.id_to_piece(spatial[k + 3]), axis="y")
        boxes.append((x1, y1, x2, y2))
    return boxes
