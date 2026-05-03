"""Phase G2 tests: ``vista-ocr eval`` metric-block dispatch + bbox
expansion asymmetry + manifest homogeneity.

Test inventory:

* ``test_back_compat_no_bboxes_keeps_old_keys`` -- the load-bearing
  regression: a manifest WITHOUT ``bboxes`` produces a sidecar with
  exactly the pre-G2 key set; no ``detection`` block leaks. Catches
  schema break.
* ``test_with_bboxes_adds_detection_block`` -- a manifest with
  ``bboxes`` produces the detection block populated with sane keys.
* ``test_bbox_expand_asymmetry_*`` -- four asymmetry tests pinning
  that --bbox-expand-px expands PREDICTIONS only:
    A. expand=0; perfect pred matches GT exactly -> F1=1.0
    B. preds 1px tighter than GT; expand=0 -> F1<1.0
    C. preds 1px tighter than GT; expand=1 -> F1=1.0 (compensated)
    D. EVEN if expand is large, F1 cannot exceed 1.0 (the cap test
       that catches "GT got expanded too" by inversion).
* ``test_heterogeneous_manifest_rejected`` -- a manifest where
  record 1 has bboxes and record 2 does not exits with a clear error.
* ``test_cer_ap_thresholds_block_optional`` -- the CER-AP block is
  populated only when --cer-ap-thresholds is set; absent otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent


def _img(path: Path, val: int = 255, size=(128, 128)) -> None:
    Image.new("L", size, val).save(path)


@pytest.fixture(scope="module")
def cli_eval_g2_env(tmp_path_factory):
    """Same shape as ``cli_eval_env`` in test_cli_eval_e2e but built
    once for G2-specific tests. Real ckpt + tokenizer; the manifests
    differ per test."""
    from vista_ocr.models.decoder import small_random_decoder
    from vista_ocr.models.encoder import FCNEncoderWidther
    from vista_ocr.models.vista_ocr import VistaOCR
    from vista_ocr.tokenizer.build_spm import train_spm
    from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
    from vista_ocr.tokenizer.tokenizer import (
        VistaTokenizer,
        list_special_and_spatial_tokens,
    )
    from vista_ocr.training.callbacks import save_checkpoint

    base = tmp_path_factory.mktemp("cli_eval_g2")
    spm_dir = base / "spm"
    spm_dir.mkdir()
    corpus = spm_dir / "c.txt"
    corpus.write_text(
        ("hello world FOO BAR baz qux quux corge\n"
         "abc def ghi jkl mno pqr stu vwx yz\n"
         "the quick brown fox jumps over the lazy dog\n"
         "Sphinx of black quartz judge my vow\n") * 200,
        encoding="utf-8",
    )
    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    train_spm(corpus, spm_dir / "tr", vocab_size=800,
              user_symbols=list_special_and_spatial_tokens(grid))
    spm_path = spm_dir / "tr.model"
    tokenizer = VistaTokenizer(spm_model_path=str(spm_path), grid=grid)

    encoder = FCNEncoderWidther(
        input_channels=1, dropout=0.0, gradient_checkpointing=False,
    )
    decoder = small_random_decoder(
        vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=4, n_heads=16,
        ffn_dim=4096, max_position_embeddings=4096,
    )
    model = VistaOCR(encoder=encoder, decoder=decoder)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    ckpt_path = base / "ckpt.pt"
    save_checkpoint(
        ckpt_path,
        step=4242, model=model, optimizer=opt,
        best_val_loss=0.5,
        extra={},
    )
    return {"spm": spm_path, "ckpt": ckpt_path, "out_dir": base}


def _canned_decode_factory(canned_lines):
    """Build a stand-in for ocr_with_layout that returns a fixed list
    of Line objects. The list lets each test control exactly what
    'predictions' look like (matching GT, smaller, larger, etc.)."""
    from vista_ocr.tokenizer.tokenizer import Line

    def _canned(_model, _image, _tokenizer, _cfg):
        return [Line(text=t, bbox=tuple(b)) for t, b in canned_lines]
    return _canned


def _run_eval(env: dict, manifest: Path, *, extra_args=()) -> dict:
    from vista_ocr.cli import main as cli_main

    out_json = manifest.with_suffix(".result.json")
    rc = cli_main([
        "eval",
        "--manifest", str(manifest),
        "--ckpt", str(env["ckpt"]),
        "--spm", str(env["spm"]),
        "--out-json", str(out_json),
        "--max-new-tokens", "4",
        *extra_args,
    ])
    assert rc == 0
    return json.loads(out_json.read_text())


def _write_manifest(path: Path, records: list[dict], img_dir: Path):
    """Write a JSONL manifest where each record's image is created on
    disk in img_dir (whitebox 128x128 by default)."""
    img_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(records):
            img_name = rec["image"]
            img_path = img_dir / img_name
            if not img_path.exists():
                _img(img_path, val=255 - (i * 30) % 200)
            fh.write(json.dumps(rec) + "\n")


# ---------- Back-compat snapshot ---------------------------------------


def test_back_compat_no_bboxes_keeps_old_keys(cli_eval_g2_env, tmp_path):
    """Pre-G2 sidecar keys must be preserved exactly when the manifest
    has no bboxes. No detection / cer_ap blocks leak in."""
    manifest = tmp_path / "no_bboxes.jsonl"
    _write_manifest(manifest, [
        {"image": "001.jpg", "ref": "hello world"},
        {"image": "002.jpg", "ref": "foo bar"},
    ], tmp_path)

    canned = _canned_decode_factory([("hello", (10, 10, 50, 30))])
    with patch("vista_ocr.inference.generate.ocr_with_layout", side_effect=canned):
        result = _run_eval(cli_eval_g2_env, manifest)

    expected_keys = {
        "ckpt", "ckpt_step", "manifest", "n_docs", "n_empty",
        "cer", "wer", "precision", "recall", "word_f1",
        "elapsed_s", "max_new_tokens", "repetition_penalty",
        "no_repeat_ngram_size",
    }
    assert set(result.keys()) == expected_keys, (
        f"sidecar key set drifted: missing={expected_keys - set(result.keys())} "
        f"extra={set(result.keys()) - expected_keys}"
    )
    assert "detection" not in result
    assert "cer_ap" not in result


# ---------- Detection block populated when bboxes present --------------


def test_with_bboxes_adds_detection_block(cli_eval_g2_env, tmp_path):
    manifest = tmp_path / "with_bboxes.jsonl"
    _write_manifest(manifest, [
        {"image": "001.jpg", "ref": "FOO BAR",
         "bboxes": [[10, 10, 50, 30, "FOO"], [60, 10, 100, 30, "BAR"]]},
        {"image": "002.jpg", "ref": "BAZ QUX",
         "bboxes": [[10, 10, 50, 30, "BAZ"], [60, 10, 100, 30, "QUX"]]},
    ], tmp_path)

    canned = _canned_decode_factory([
        ("FOO", (10, 10, 50, 30)), ("BAR", (60, 10, 100, 30)),
    ])
    with patch("vista_ocr.inference.generate.ocr_with_layout", side_effect=canned):
        result = _run_eval(cli_eval_g2_env, manifest)

    assert "detection" in result
    det = result["detection"]
    for k in ("deteval_precision", "deteval_recall", "deteval_f1",
              "area_f1", "ap_at_iou", "iou_threshold", "bbox_expand_px"):
        assert k in det, f"detection block missing key {k}"
    assert det["bbox_expand_px"] == 0   # default
    # AP@IoU 0.5 should be perfect since canned pred boxes match GT exactly.
    assert det["ap_at_iou"]["0.5"] == pytest.approx(1.0)


# ---------- Bbox-expand asymmetry (the 4-test contract) ---------------


def test_bbox_expand_A_perfect_match_no_expand(cli_eval_g2_env, tmp_path):
    """expand=0 + perfect pred = perfect F1."""
    manifest = tmp_path / "exp_A.jsonl"
    _write_manifest(manifest, [
        {"image": "001.jpg", "ref": "FOO",
         "bboxes": [[10, 10, 50, 30, "FOO"]]},
    ], tmp_path)
    canned = _canned_decode_factory([("FOO", (10, 10, 50, 30))])
    with patch("vista_ocr.inference.generate.ocr_with_layout", side_effect=canned):
        result = _run_eval(cli_eval_g2_env, manifest, extra_args=["--bbox-expand-px", "0"])
    assert result["detection"]["deteval_f1"] == pytest.approx(1.0)


def test_bbox_expand_B_tighter_pred_no_expand_underscores(cli_eval_g2_env, tmp_path):
    """Pred 2px tighter than GT on every side; expand=0; deteval F1
    drops below 1.0 because intersection/gt_area falls below tr=0.8."""
    manifest = tmp_path / "exp_B.jsonl"
    _write_manifest(manifest, [
        {"image": "001.jpg", "ref": "FOO",
         "bboxes": [[10, 10, 50, 30, "FOO"]]},
    ], tmp_path)
    # Pred 2px in on every side: (12,12,48,28). Area = 36*16 = 576;
    # GT area = 40*20 = 800; intersection = 576; tr = 576/800 = 0.72 < 0.8.
    canned = _canned_decode_factory([("FOO", (12, 12, 48, 28))])
    with patch("vista_ocr.inference.generate.ocr_with_layout", side_effect=canned):
        result = _run_eval(cli_eval_g2_env, manifest, extra_args=["--bbox-expand-px", "0"])
    assert result["detection"]["deteval_f1"] < 1.0


def test_bbox_expand_C_tighter_pred_compensated(cli_eval_g2_env, tmp_path):
    """Same tighter-pred case, but --bbox-expand-px=2 expands the
    prediction back to GT size -> F1 returns to 1.0."""
    manifest = tmp_path / "exp_C.jsonl"
    _write_manifest(manifest, [
        {"image": "001.jpg", "ref": "FOO",
         "bboxes": [[10, 10, 50, 30, "FOO"]]},
    ], tmp_path)
    canned = _canned_decode_factory([("FOO", (12, 12, 48, 28))])
    with patch("vista_ocr.inference.generate.ocr_with_layout", side_effect=canned):
        result = _run_eval(cli_eval_g2_env, manifest, extra_args=["--bbox-expand-px", "2"])
    assert result["detection"]["deteval_f1"] == pytest.approx(1.0)


def test_bbox_expand_D_huge_expand_cannot_inflate_above_one(cli_eval_g2_env, tmp_path):
    """The asymmetry contract: GT is NEVER expanded. With a huge
    expand value, the predicted box becomes MUCH bigger than GT
    (precision drops because pred area >> intersection), so F1 must
    fall, not exceed 1.0. If GT were also expanded, F1 could stay
    near 1.0 at any expand value -- this test would catch that."""
    manifest = tmp_path / "exp_D.jsonl"
    _write_manifest(manifest, [
        {"image": "001.jpg", "ref": "FOO",
         "bboxes": [[10, 10, 50, 30, "FOO"]]},
    ], tmp_path)
    canned = _canned_decode_factory([("FOO", (10, 10, 50, 30))])
    with patch("vista_ocr.inference.generate.ocr_with_layout", side_effect=canned):
        result = _run_eval(cli_eval_g2_env, manifest, extra_args=["--bbox-expand-px", "50"])
    # With +50px expand on a 40x20 box, pred becomes 140x120 = 16800;
    # GT is 40x20 = 800; intersection = 800. tp = 800/16800 = 0.048,
    # which is below tp=0.4 -> no match -> F1 drops.
    assert result["detection"]["deteval_f1"] < 1.0


# ---------- Heterogeneity rejection ------------------------------------


def test_heterogeneous_manifest_rejected(cli_eval_g2_env, tmp_path):
    """A manifest where record 1 has bboxes and record 2 does not is
    a smell that hides 'I forgot to annotate half my data.' Reject
    explicitly."""
    manifest = tmp_path / "hetero.jsonl"
    _write_manifest(manifest, [
        {"image": "001.jpg", "ref": "FOO",
         "bboxes": [[10, 10, 50, 30, "FOO"]]},
        {"image": "002.jpg", "ref": "BAR"},   # missing bboxes
    ], tmp_path)

    canned = _canned_decode_factory([("FOO", (10, 10, 50, 30))])
    with patch("vista_ocr.inference.generate.ocr_with_layout", side_effect=canned):
        with pytest.raises(SystemExit, match="heterogeneous"):
            _run_eval(cli_eval_g2_env, manifest)


# ---------- CER-AP block optional --------------------------------------


def test_cer_ap_block_absent_by_default(cli_eval_g2_env, tmp_path):
    manifest = tmp_path / "no_cerap.jsonl"
    _write_manifest(manifest, [
        {"image": "001.jpg", "ref": "FOO"},
    ], tmp_path)
    canned = _canned_decode_factory([("FOO", (0, 0, 0, 0))])
    with patch("vista_ocr.inference.generate.ocr_with_layout", side_effect=canned):
        result = _run_eval(cli_eval_g2_env, manifest)
    assert "cer_ap" not in result


def test_cer_ap_block_populated_when_thresholds_passed(cli_eval_g2_env, tmp_path):
    manifest = tmp_path / "with_cerap.jsonl"
    _write_manifest(manifest, [
        {"image": "001.jpg", "ref": "FOO"},
    ], tmp_path)
    canned = _canned_decode_factory([("FOO", (0, 0, 0, 0))])
    with patch("vista_ocr.inference.generate.ocr_with_layout", side_effect=canned):
        result = _run_eval(
            cli_eval_g2_env, manifest,
            extra_args=["--cer-ap-thresholds", "0.0,0.1,0.3"],
        )
    assert "cer_ap" in result
    cer_ap = result["cer_ap"]
    assert sorted(cer_ap["ap"].keys()) == ["0.00", "0.10", "0.30"]
    # Perfect prediction -> AP=1.0 at every threshold.
    assert cer_ap["ap"]["0.00"] == pytest.approx(1.0)
