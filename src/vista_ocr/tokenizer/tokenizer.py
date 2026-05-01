"""VistaTokenizer: SentencePiece subwords + special tokens + spatial tokens.

Implements the three serialization schemes from VISTA-OCR Table 7
("Original", "Segmented", "Unified") and the prompt formats from Figs. 6/7
("Read at x1,y1,x2,y2" for region-OCR and "<find_it> {text}" for content-
based localization).

Per the paper, coordinate values inside *prompts* are literal numeric text
(subword-tokenized as digits), while coordinate values in the *output*
sequence are emitted as spatial tokens (<x_i>, <y_j>). This module is the
single source of truth for both directions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import sentencepiece as spm

from vista_ocr.tokenizer.spatial_tokens import SpatialGrid

# --- Special tokens (also see configs/base.yaml tokenizer.special_tokens) ---
SPECIAL_TOKENS: list[str] = [
    "<task=ocr>",
    "<task=ocr_layout>",
    "<task=region_ocr>",
    "<task=find_it>",
    "</text>",
    "</location>",
    "<find_it>",
]

PAD = "<pad>"
UNK = "<unk>"
BOS = "<s>"
EOS = "</s>"


@dataclass
class Line:
    """One layout line: bounding box (px on the page canvas) + text."""

    text: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels

    def sort_key(self) -> tuple[int, int]:
        # Top-left to bottom-right ordering used by VISTA_omni.
        return (self.bbox[1], self.bbox[0])


def list_special_and_spatial_tokens(grid: SpatialGrid) -> list[str]:
    """All non-subword tokens — pass to SentencePiece as user_defined_symbols
    when training the SPM model."""
    return SPECIAL_TOKENS + grid.all_tokens()


# Matches any spatial token: <x_123>, <y_45>, or <xy_67>.
_SPATIAL_RE = re.compile(r"<(?:x|y|xy)_\d+>")


class VistaTokenizer:
    def __init__(self, spm_model_path: str | Path, grid: SpatialGrid) -> None:
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(spm_model_path))
        self.grid = grid

        # Cache id sets for fast classification at loss time.
        self._spatial_token_strs: set[str] = set(grid.all_tokens())
        self._special_token_strs: set[str] = set(SPECIAL_TOKENS)
        self._spatial_ids: set[int] = self._ids_for(self._spatial_token_strs)
        self._special_ids: set[int] = self._ids_for(self._special_token_strs)

        self.pad_id = self.sp.piece_to_id(PAD)
        self.unk_id = self.sp.piece_to_id(UNK)
        self.bos_id = self.sp.piece_to_id(BOS)
        self.eos_id = self.sp.piece_to_id(EOS)

    # --- vocab / id queries ----------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    def _ids_for(self, pieces: Iterable[str]) -> set[int]:
        ids: set[int] = set()
        for p in pieces:
            i = self.sp.piece_to_id(p)
            if i != self.unk_id_or_default():
                ids.add(i)
        return ids

    def unk_id_or_default(self) -> int:
        # SentencePiece returns the unk id for any unknown piece. We compare
        # against the model's actual unk id when checking presence above.
        return self.sp.piece_to_id(UNK)

    def is_spatial_id(self, token_id: int) -> bool:
        return token_id in self._spatial_ids

    def is_special_id(self, token_id: int) -> bool:
        return token_id in self._special_ids

    def piece_to_id(self, piece: str) -> int:
        return self.sp.piece_to_id(piece)

    def id_to_piece(self, token_id: int) -> str:
        return self.sp.id_to_piece(token_id)

    # --- encode helpers --------------------------------------------------

    def encode_text(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.sp.EncodeAsIds(text)
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def encode_pieces(self, pieces: Iterable[str]) -> list[int]:
        # Spatial / special tokens are user_defined_symbols and must round-trip
        # exactly. Subwords are handled by SentencePiece for everything else.
        return [self.sp.piece_to_id(p) for p in pieces]

    def decode_ids(self, ids: Iterable[int]) -> str:
        # Decoding via DecodeIds keeps user_defined_symbols intact verbatim.
        return self.sp.DecodeIds(list(ids))

    # --- serialization (paper Table 7) -----------------------------------

    def serialize_lines(self, lines: list[Line], scheme: str | None = None) -> list[int]:
        """Convert a list of layout-aware Line objects into a sequence of
        token ids using the requested encoding scheme.

        Lines are sorted top-left to bottom-right (VISTA_omni reading order)
        before serialization.
        """
        scheme = scheme or self.grid.scheme
        ordered = sorted(lines, key=Line.sort_key)
        if scheme == "original":
            return self._serialize_original(ordered)
        if scheme == "segmented":
            return self._serialize_segmented(ordered)
        if scheme == "unified":
            return self._serialize_unified(ordered)
        raise ValueError(f"Unknown scheme: {scheme}")

    def _serialize_original(self, lines: list[Line]) -> list[int]:
        # <x1><y1> w1 w2 ... wm <x2><y2>  per line, concatenated.
        out: list[int] = []
        for line in lines:
            x1, y1, x2, y2 = line.bbox
            out.append(self.sp.piece_to_id(self.grid.x_token(x1)))
            out.append(self.sp.piece_to_id(self.grid.y_token(y1)))
            out.extend(self.encode_text(line.text))
            out.append(self.sp.piece_to_id(self.grid.x_token(x2)))
            out.append(self.sp.piece_to_id(self.grid.y_token(y2)))
        return out

    def _serialize_segmented(self, lines: list[Line]) -> list[int]:
        # All text first, then </text>, then all coords, then </location>.
        out: list[int] = []
        for line in lines:
            out.extend(self.encode_text(line.text))
        out.append(self.sp.piece_to_id("</text>"))
        for line in lines:
            x1, y1, x2, y2 = line.bbox
            out.append(self.sp.piece_to_id(self.grid.x_token(x1)))
            out.append(self.sp.piece_to_id(self.grid.y_token(y1)))
            out.append(self.sp.piece_to_id(self.grid.x_token(x2)))
            out.append(self.sp.piece_to_id(self.grid.y_token(y2)))
        out.append(self.sp.piece_to_id("</location>"))
        return out

    def _serialize_unified(self, lines: list[Line]) -> list[int]:
        # Single XY token set: <xy_a><xy_b> w... <xy_c><xy_d>
        n = max(self.grid.n_x, self.grid.n_y)
        q = self.grid.quantizer_px

        def xy(px: int) -> str:
            return f"<xy_{min(n - 1, max(0, px // q))}>"

        out: list[int] = []
        for line in lines:
            x1, y1, x2, y2 = line.bbox
            out.append(self.sp.piece_to_id(xy(x1)))
            out.append(self.sp.piece_to_id(xy(y1)))
            out.extend(self.encode_text(line.text))
            out.append(self.sp.piece_to_id(xy(x2)))
            out.append(self.sp.piece_to_id(xy(y2)))
        return out

    # --- prompt builders (paper Figs. 6 / 7) -----------------------------

    def build_region_ocr_prompt(self, bbox: tuple[int, int, int, int]) -> list[int]:
        """Region-Based OCR. Paper Fig. 6: literal text "Read at x1,y1,x2,y2".
        The coordinates are subword-tokenized digits, *not* spatial tokens —
        spatial tokens only appear in the output."""
        x1, y1, x2, y2 = bbox
        prompt = f"Read at {x1},{y1},{x2},{y2}"
        return [
            self.sp.piece_to_id("<task=region_ocr>"),
            *self.encode_text(prompt),
        ]

    def build_find_it_prompt(self, query_text: str) -> list[int]:
        """Content-Based Localization. Paper Fig. 7: '<find_it> {text}'."""
        return [
            self.sp.piece_to_id("<task=find_it>"),
            self.sp.piece_to_id("<find_it>"),
            *self.encode_text(query_text),
        ]

    def build_ocr_prompt(self, with_layout: bool) -> list[int]:
        return [self.sp.piece_to_id("<task=ocr_layout>" if with_layout else "<task=ocr>")]

    # --- output parser (inference) --------------------------------------

    def parse_original_output(self, ids: list[int]) -> list[Line]:
        """Inverse of `_serialize_original`: parse runs of
        <x><y> ... text ... <x><y> back into Line objects.

        Strict parser: requires alternating x,y / text / x,y triplets.
        Stops cleanly on first malformed run."""
        out: list[Line] = []
        i = 0
        n = len(ids)
        while i < n:
            tok = self.id_to_piece(ids[i])
            if not tok.startswith("<x_"):
                i += 1
                continue
            if i + 1 >= n:
                break
            x1_tok = self.id_to_piece(ids[i])
            y1_tok = self.id_to_piece(ids[i + 1])
            if not y1_tok.startswith("<y_"):
                i += 1
                continue
            j = i + 2
            text_ids: list[int] = []
            while j < n - 1:
                tj = self.id_to_piece(ids[j])
                if tj.startswith("<x_"):
                    break
                text_ids.append(ids[j])
                j += 1
            if j >= n - 1:
                break
            x2_tok = self.id_to_piece(ids[j])
            y2_tok = self.id_to_piece(ids[j + 1])
            if not (x2_tok.startswith("<x_") and y2_tok.startswith("<y_")):
                i = j
                continue
            text = self.sp.DecodeIds(text_ids)
            bbox = (
                self._token_to_px(x1_tok, axis="x"),
                self._token_to_px(y1_tok, axis="y"),
                self._token_to_px(x2_tok, axis="x"),
                self._token_to_px(y2_tok, axis="y"),
            )
            out.append(Line(text=text, bbox=bbox))
            i = j + 2
        return out

    def _token_to_px(self, tok: str, axis: str) -> int:
        m = _SPATIAL_RE.fullmatch(tok)
        if not m:
            raise ValueError(f"Not a spatial token: {tok!r}")
        idx = int(tok.split("_")[1].rstrip(">"))
        return idx * self.grid.quantizer_px
