"""Unit tests for VistaTokenizer.

We train a tiny in-memory SPM on a synthetic corpus so tests run quickly
without any downloads."""
from __future__ import annotations

from pathlib import Path

import pytest

from vista_ocr.tokenizer.build_spm import train_spm
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import (
    SPECIAL_TOKENS,
    Line,
    VistaTokenizer,
    list_special_and_spatial_tokens,
)

CORPUS = """\
The quick brown fox jumps over the lazy dog.
Sphinx of black quartz, judge my vow.
Pack my box with five dozen liquor jugs.
How vexingly quick daft zebras jump!
The five boxing wizards jump quickly.
Read at coordinates and find_it returns boxes.
Receipt total: $42.00. Subtotal: $39.99. Tax: $2.01.
Invoice number 1234 dated 2024-12-31 for ACME Corp.
Digits sample: 0 1 2 3 4 5 6 7 8 9. Pairs: 56 67 78 89 90.
Coordinates 66,184,97,186 and 250,300,400,500.
"""


@pytest.fixture(scope="module")
def grid() -> SpatialGrid:
    # Small canvas keeps the spatial vocab manageable for the tiny SPM.
    return SpatialGrid(canvas_h=200, canvas_w=200, quantizer_px=10, scheme="original")


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory: pytest.TempPathFactory, grid: SpatialGrid) -> VistaTokenizer:
    tmp = tmp_path_factory.mktemp("spm")
    corpus_path = tmp / "tiny_corpus.txt"
    # Repeat to give SPM enough material at vocab_size=300.
    corpus_path.write_text(CORPUS * 200, encoding="utf-8")
    out_prefix = tmp / "tiny"
    train_spm(
        corpus_path=corpus_path,
        out_prefix=out_prefix,
        vocab_size=300,
        user_symbols=list_special_and_spatial_tokens(grid),
        character_coverage=1.0,
    )
    return VistaTokenizer(spm_model_path=out_prefix.with_suffix(".model"), grid=grid)


def test_special_tokens_in_vocab(tokenizer: VistaTokenizer) -> None:
    for tok in SPECIAL_TOKENS:
        i = tokenizer.piece_to_id(tok)
        assert i != tokenizer.unk_id, f"Special token {tok} mapped to <unk>"
        assert tokenizer.is_special_id(i)


def test_spatial_tokens_in_vocab(tokenizer: VistaTokenizer, grid: SpatialGrid) -> None:
    for tok in grid.all_tokens():
        i = tokenizer.piece_to_id(tok)
        assert i != tokenizer.unk_id, f"Spatial token {tok} mapped to <unk>"
        assert tokenizer.is_spatial_id(i)


def test_serialize_original_round_trip(tokenizer: VistaTokenizer) -> None:
    lines = [
        Line(text="hello world", bbox=(10, 20, 100, 40)),
        Line(text="goodbye", bbox=(10, 60, 80, 80)),
    ]
    ids = tokenizer.serialize_lines(lines, scheme="original")
    parsed = tokenizer.parse_original_output(ids)
    assert len(parsed) == 2
    # Reading-order sort preserves input order here (already TLBR).
    assert parsed[0].bbox == (10, 20, 100, 40)
    assert parsed[1].bbox == (10, 60, 80, 80)
    # SentencePiece round-trip preserves text up to whitespace normalization.
    assert "hello" in parsed[0].text
    assert "world" in parsed[0].text
    assert "goodbye" in parsed[1].text


def test_serialize_segmented_contains_separators(tokenizer: VistaTokenizer) -> None:
    lines = [Line(text="hello", bbox=(10, 20, 100, 40))]
    ids = tokenizer.serialize_lines(lines, scheme="segmented")
    pieces = [tokenizer.id_to_piece(i) for i in ids]
    assert "</text>" in pieces
    assert "</location>" in pieces
    # </text> must come before </location>
    assert pieces.index("</text>") < pieces.index("</location>")


def test_serialize_unified_uses_xy_tokens() -> None:
    grid = SpatialGrid(canvas_h=200, canvas_w=200, quantizer_px=10, scheme="unified")
    # Need a fresh tokenizer for this scheme so <xy_*> is in vocab.
    import tempfile

    from vista_ocr.tokenizer.build_spm import train_spm
    from vista_ocr.tokenizer.tokenizer import list_special_and_spatial_tokens

    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "c.txt"
        corpus.write_text(CORPUS * 200, encoding="utf-8")
        prefix = Path(tmp) / "tiny_unified"
        train_spm(
            corpus_path=corpus,
            out_prefix=prefix,
            vocab_size=300,
            user_symbols=list_special_and_spatial_tokens(grid),
        )
        tk = VistaTokenizer(spm_model_path=prefix.with_suffix(".model"), grid=grid)
        ids = tk.serialize_lines(
            [Line(text="hello", bbox=(10, 20, 100, 40))],
            scheme="unified",
        )
        pieces = [tk.id_to_piece(i) for i in ids]
        xy_tokens = [p for p in pieces if p.startswith("<xy_")]
        assert len(xy_tokens) == 4  # x1y1...x2y2


def test_reading_order_top_left_to_bottom_right(tokenizer: VistaTokenizer) -> None:
    # Lines provided out-of-order on purpose.
    lines = [
        Line(text="bottom", bbox=(10, 100, 100, 120)),
        Line(text="top", bbox=(10, 10, 100, 30)),
        Line(text="middle", bbox=(10, 50, 100, 70)),
    ]
    ids = tokenizer.serialize_lines(lines, scheme="original")
    parsed = tokenizer.parse_original_output(ids)
    ys = [p.bbox[1] for p in parsed]
    assert ys == sorted(ys)


def test_region_ocr_prompt_is_literal_text(tokenizer: VistaTokenizer) -> None:
    """Paper Fig. 6: prompt is 'Read at x1,y1,x2,y2' — coords are digits, not
    spatial tokens. Verify no <x_*>/<y_*> tokens leak into the prompt."""
    ids = tokenizer.build_region_ocr_prompt(bbox=(66, 184, 97, 186))
    pieces = [tokenizer.id_to_piece(i) for i in ids]
    # Task token at start.
    assert pieces[0] == "<task=region_ocr>"
    # No spatial tokens anywhere.
    assert not any(tokenizer.is_spatial_id(i) for i in ids)
    # The literal text must round-trip.
    text = tokenizer.decode_ids(ids[1:]).replace(" ", "")
    assert "Readat66,184,97,186" in text


def test_find_it_prompt_format(tokenizer: VistaTokenizer) -> None:
    ids = tokenizer.build_find_it_prompt("hello world")
    pieces = [tokenizer.id_to_piece(i) for i in ids]
    assert pieces[0] == "<task=find_it>"
    assert pieces[1] == "<find_it>"


def test_classify_text_vs_spatial_for_loss(tokenizer: VistaTokenizer) -> None:
    line = Line(text="hello", bbox=(10, 20, 100, 40))
    ids = tokenizer.serialize_lines([line], scheme="original")
    spatial_count = sum(1 for i in ids if tokenizer.is_spatial_id(i))
    assert spatial_count == 4  # x1, y1, x2, y2
    text_count = sum(1 for i in ids if not tokenizer.is_spatial_id(i))
    assert text_count >= 1
