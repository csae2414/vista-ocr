"""Tests for data preprocessing, collation, mixture sampler, and the
synthetic generators."""
from __future__ import annotations

import pytest

from vista_ocr.data.collate import build_target_ids, collate
from vista_ocr.data.mixture import MixedTaskStream, TaskMix
from vista_ocr.data.preprocess import (
    PreprocessConfig,
    is_latin_text,
    pad_to_multiple,
    resize_to_canvas,
)
from vista_ocr.data.synth.sroie_synth import SroieSynthConfig
from vista_ocr.data.synth.sroie_synth import generate_sample as gen_sroie
from vista_ocr.data.synth.synthdog_bbox import SynthDogConfig
from vista_ocr.data.synth.synthdog_bbox import generate_sample as gen_synthdog
from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.build_spm import train_spm
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import (
    Line,
    VistaTokenizer,
    list_special_and_spatial_tokens,
)

# ---------- shared tokenizer fixture ----------

CORPUS = (
    "The quick brown fox jumps over the lazy dog.\n"
    "Pack my box with five dozen liquor jugs.\n"
    "Receipt total: $42.00. Subtotal: $39.99. Tax: $2.01.\n"
    "Read at 10,20,100,40 and find_it returns boxes.\n"
    "Digits: 0 1 2 3 4 5 6 7 8 9.\n"
)


@pytest.fixture(scope="module")
def grid() -> SpatialGrid:
    return SpatialGrid(canvas_h=1024, canvas_w=768, quantizer_px=10, scheme="original")


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory: pytest.TempPathFactory, grid: SpatialGrid) -> VistaTokenizer:
    tmp = tmp_path_factory.mktemp("spm_data")
    corpus_path = tmp / "tiny.txt"
    corpus_path.write_text(CORPUS * 200, encoding="utf-8")
    out = tmp / "data_spm"
    train_spm(corpus_path, out, vocab_size=400, user_symbols=list_special_and_spatial_tokens(grid))
    return VistaTokenizer(out.with_suffix(".model"), grid)


# ---------- preprocess ----------

def test_is_latin_text():
    assert is_latin_text("Hello world 123")
    assert not is_latin_text("こんにちは")


def test_resize_smaller_image_no_downscale():
    from PIL import Image

    img = Image.new("L", (100, 100), 255)
    out, scale, _ = resize_to_canvas(img, PreprocessConfig(target_h=3508, target_w=2480))
    assert scale == 1.0
    assert out.size == (100, 100)


def test_resize_oversize_image_downscales():
    from PIL import Image

    img = Image.new("L", (5000, 5000), 255)
    out, scale, (h, w) = resize_to_canvas(img, PreprocessConfig(target_h=2000, target_w=2000))
    assert scale < 1.0
    assert out.size == (w, h)
    assert max(h, w) <= 2000


def test_pad_to_multiple():
    from PIL import Image

    img = Image.new("L", (50, 70), 255)
    out, (dh, dw) = pad_to_multiple(img, multiple=32)
    assert out.size[0] % 32 == 0
    assert out.size[1] % 32 == 0
    assert dh > 0 and dw > 0


# ---------- collate ----------

def test_build_target_ids_layout(tokenizer: VistaTokenizer):
    sample = Sample(
        image=None,
        lines=[Line("hello", (10, 20, 100, 40))],
        task="ocr_layout",
    )
    seq, plen = build_target_ids(tokenizer, sample)
    assert seq[0] == tokenizer.bos_id
    assert seq[-1] == tokenizer.eos_id
    assert plen == 2  # bos + <task=ocr_layout>


def test_build_target_ids_region_ocr_requires_bbox(tokenizer: VistaTokenizer):
    s = Sample(image=None, lines=[Line("x", (0, 0, 1, 1))], task="region_ocr")
    with pytest.raises(ValueError):
        build_target_ids(tokenizer, s)


def test_build_target_ids_find_it_no_match_emits_no_bbox(tokenizer: VistaTokenizer):
    s = Sample(
        image=None,
        lines=[Line("hello", (10, 20, 30, 40))],
        task="find_it",
        query_text="absent",
    )
    seq, _ = build_target_ids(tokenizer, s)
    spatial_count = sum(1 for i in seq if tokenizer.is_spatial_id(i))
    assert spatial_count == 0


# ---------- truncate keyword (used by tools/audit_tokenizer.py) ----------


def _long_layout_sample(n_lines: int) -> Sample:
    """Synthesise a ``Sample`` with enough lines to blow past
    ``MAX_TARGET_TOKENS`` under ``ocr_layout`` serialisation."""
    return Sample(
        image=None,
        lines=[
            Line(f"line {i} fox jumps over lazy dog", (10 + (i % 4) * 50, 20 + i * 10, 200, 40 + i * 10))
            for i in range(n_lines)
        ],
        task="ocr_layout",
    )


def test_build_target_ids_truncates_by_default(tokenizer: VistaTokenizer):
    """Production behaviour: a Sample whose serialised target
    exceeds ``MAX_TARGET_TOKENS`` is clipped to that length."""
    from vista_ocr.data.collate import MAX_TARGET_TOKENS
    s = _long_layout_sample(n_lines=400)
    seq, _plen = build_target_ids(tokenizer, s)
    assert len(seq) == MAX_TARGET_TOKENS


def test_build_target_ids_returns_full_length_when_truncate_false(tokenizer: VistaTokenizer):
    """Audit / diagnostics path: ``truncate=False`` returns the
    pre-truncation sequence so tools can measure truncation
    pressure rather than read the post-clip ceiling."""
    from vista_ocr.data.collate import MAX_TARGET_TOKENS
    s = _long_layout_sample(n_lines=400)
    seq, _plen = build_target_ids(tokenizer, s, truncate=False)
    assert len(seq) > MAX_TARGET_TOKENS, (
        f"expected pre-truncation length > {MAX_TARGET_TOKENS}, "
        f"got {len(seq)} (did MAX_TARGET_TOKENS get raised?)"
    )
    # The full sequence must still start with bos and end with eos.
    assert seq[0] == tokenizer.bos_id
    assert seq[-1] == tokenizer.eos_id


def test_build_target_ids_short_sample_unchanged_under_truncate_false(tokenizer: VistaTokenizer):
    """Regression sanity: a Sample that does not need truncation
    produces the same sequence under ``truncate=True`` and
    ``truncate=False``. The keyword is opt-out for long pages
    only; it must not change short-page behaviour."""
    s = Sample(
        image=None,
        lines=[Line("hello", (10, 20, 100, 40))],
        task="ocr_layout",
    )
    seq_default, plen_default = build_target_ids(tokenizer, s)
    seq_no_trunc, plen_no_trunc = build_target_ids(tokenizer, s, truncate=False)
    assert seq_default == seq_no_trunc
    assert plen_default == plen_no_trunc


def test_collate_pads_to_max_in_batch(tokenizer: VistaTokenizer):
    from PIL import Image

    s1 = Sample(
        image=Image.new("L", (32, 64), 255),
        lines=[Line("a", (0, 0, 8, 16))],
        task="ocr",
    )
    s2 = Sample(
        image=Image.new("L", (48, 32), 255),
        lines=[Line("ab cd", (0, 0, 16, 16))],
        task="ocr_layout",
    )
    batch = collate([s1, s2], tokenizer, PreprocessConfig(pad_multiple=16, target_h=3508, target_w=2480))
    assert batch.images.shape[0] == 2
    assert batch.decoder_input_ids.shape == batch.labels.shape
    assert batch.images.shape[-2] % 16 == 0 and batch.images.shape[-1] % 16 == 0
    # Both samples should have prompt-mask True at position 0 (bos).
    assert batch.prompt_mask[:, 0].all()


# ---------- Phase 3: bbox-scaling on resize ----------

def test_collate_scales_bboxes_when_image_is_resized(tokenizer: VistaTokenizer):
    """Phase 3: when ``resize_to_canvas`` shrinks the image, bboxes
    must be scaled proportionally so they stay inside the canvas.
    Catches the exact regression we hit when augment was enabled at
    page_preset=large with PDFA pages — Albumentations rejected
    bboxes whose normalised coords exceeded 1.0 because the bbox
    coords still referenced the original (larger) image."""
    from PIL import Image

    # Original image 1500x800; bbox spans the whole image. Resize
    # canvas 750x400 (exactly 0.5x scale on both axes). After scaling
    # the bbox should be exactly half size.
    img = Image.new("L", (1500, 800), 255)
    s = Sample(
        image=img,
        lines=[Line("hello", (0, 0, 1500, 800))],
        task="ocr_layout",
    )
    pre_cfg = PreprocessConfig(target_h=400, target_w=750, pad_multiple=32)
    batch = collate([s], tokenizer, pre_cfg)
    # The image must have been resized; key check is no exception
    # was raised AND the resulting image is no larger than canvas.
    assert batch.images.shape[-1] <= 768   # 750 padded to multiple of 32 = 768
    assert batch.images.shape[-2] <= 416   # 400 padded to multiple of 32 = 416


def test_collate_with_augment_does_not_overflow_canvas(tokenizer: VistaTokenizer):
    """Phase 3 regression test: with augment enabled and a sample
    whose original-image bbox would overflow the resized canvas,
    Albumentations must NOT raise ``Expected x_max ... in [0, 1]``.

    Construct the exact failure: original 1500x800 image, bbox at the
    far right edge, target canvas 1100x850. Without bbox-scaling, the
    bbox at x=1400-1500 in original coords exceeds the resized 1100
    width once Albumentations normalises. With bbox-scaling, it stays
    inside.
    """
    pytest.importorskip("albumentations")
    from PIL import Image
    from vista_ocr.data.augment import AugmentConfig

    img = Image.new("L", (1500, 800), 255)
    s = Sample(
        image=img,
        lines=[Line("rightmost", (1400, 100, 1490, 200))],
        task="ocr_layout",
    )
    pre_cfg = PreprocessConfig(
        target_h=850, target_w=1100, pad_multiple=32,
        augment=AugmentConfig(
            enabled=True, rotate_deg=0.0,
            brightness_limit=0.0, contrast_limit=0.0,
            blur_max_sigma=0.0, jpeg_quality_min=95, jpeg_quality_max=95,
            p_each=0.0,
        ),
    )
    # Should not raise.
    batch = collate([s], tokenizer, pre_cfg)
    assert batch.images.shape[0] == 1


# ---------- mixture ----------

def test_mixture_relabels_to_chosen_task():
    base = [
        Sample(image=None, lines=[Line("hello world foo", (0, 0, 50, 20))], task="ocr_layout"),
        Sample(image=None, lines=[Line("the quick brown fox", (0, 0, 80, 20))], task="ocr_layout"),
    ]
    mix = TaskMix({"ocr": 0.0, "ocr_layout": 0.0, "region_ocr": 1.0, "find_it": 0.0})
    stream = MixedTaskStream(iter(base), mix, seed=0)
    out = list(stream)
    assert all(s.task == "region_ocr" for s in out)
    assert all(s.query_bbox is not None for s in out)


def test_taskmix_weights_govern_sampling_distribution():
    """A3 gap closed: passing non-uniform --w-* weights to stage3 must
    actually skew the per-step task distribution. Catches the class of
    bug where weights are accepted but ignored downstream."""
    import random
    mix = TaskMix({
        "ocr": 0.7, "ocr_layout": 0.1, "region_ocr": 0.1, "find_it": 0.1,
    })
    rng = random.Random(0)
    counts = {"ocr": 0, "ocr_layout": 0, "region_ocr": 0, "find_it": 0}
    for _ in range(2000):
        counts[mix.sample_task(rng)] += 1
    # OCR should dominate (~70%) and others should be small but present.
    assert counts["ocr"] > 1200    # ~70% of 2000 ± stochastic
    assert counts["ocr"] < 1600
    for other in ("ocr_layout", "region_ocr", "find_it"):
        assert 100 < counts[other] < 350  # ~10%
    # And degenerate weight=0 truly excludes a task.
    rng2 = random.Random(0)
    mix_zero = TaskMix({"ocr": 1.0, "ocr_layout": 0.0,
                         "region_ocr": 0.0, "find_it": 0.0})
    samples = [mix_zero.sample_task(rng2) for _ in range(500)]
    assert all(s == "ocr" for s in samples)


# ---------- synth ----------

def test_synthdog_emits_bboxes_within_canvas():
    sample = gen_synthdog(["hello", "world"], SynthDogConfig(canvas_h=128, canvas_w=128, seed=1))
    assert len(sample.lines) <= 2
    for line in sample.lines:
        x1, y1, x2, y2 = line.bbox
        assert 0 <= x1 <= x2 <= 128
        assert 0 <= y1 <= y2 <= 128


def test_sroie_synth_produces_typical_receipt_lines():
    sample = gen_sroie(SroieSynthConfig(
        canvas_h=512, canvas_w=384, seed=42,
        # Disable augmentations so the bbox layout stays predictable.
        blur_prob=0.0, background_markup_prob=0.0, slant_prob=0.0,
        shadow_prob=0.0, poor_resolution_prob=0.0,
    ))
    texts = [ln.text for ln in sample.lines]
    assert any("TOTAL" in t for t in texts)
    assert any("SUBTOTAL" in t for t in texts)


def test_sroie_synth_with_full_augmentations():
    """Augmentations must not crash, but they may shift bboxes. Just make
    sure we still get some lines."""
    sample = gen_sroie(SroieSynthConfig(
        canvas_h=512, canvas_w=384, seed=7,
        blur_prob=1.0, background_markup_prob=1.0, slant_prob=1.0,
        shadow_prob=1.0, poor_resolution_prob=1.0,
    ))
    assert len(sample.lines) > 3
