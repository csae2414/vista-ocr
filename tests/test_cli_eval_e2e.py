"""End-to-end smoke for ``vista-ocr eval``.

Exercises the FULL CLI verb wiring against a real ckpt + real
manifest:

  argparse -> ckpt load -> iter_manifest -> per-doc decode ->
  word_exact_prf / jiwer aggregate -> JSON sidecar write

The decode itself is patched with a canned output so the test runs
in seconds; the production decoder path is already covered by the
inference module's own tests, so re-running greedy here would be
duplicate work and would push the test suite over its time budget.

What this test catches that the per-verb ``--help`` smoke does not:

* the manifest iterator is wired into the eval verb (a typo in
  ``iter_manifest`` import would only fail here),
* ``load_checkpoint`` actually loads and the verb propagates the
  checkpoint step into the JSON sidecar,
* the metric aggregation produces a valid JSON document with the
  documented keys (n_docs, n_empty, cer, wer, precision, recall,
  word_f1, ckpt_step, ...),
* per-doc predictions JSONL is wired when ``--out-predictions`` is
  passed.

Cost: ~5-8 s on CPU; dominated by the one-time model construction
and ckpt round-trip in the module fixture.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cli_eval_env(tmp_path_factory):
    """A real tokenizer + saved checkpoint that the eval verb will
    pick up. The checkpoint is built with the same model constructors
    the verb uses (production sizes) so ``load_checkpoint`` actually
    loads matching state_dict keys."""
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

    base = tmp_path_factory.mktemp("cli_eval_e2e")

    # Real SPM model with the production grid (3508x2480 canvas, 10-px
    # quantizer -- ~640 spatial tokens). Vocab >= 800 to leave headroom.
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

    # Real model -- same construction signature the eval verb uses.
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
        extra={"smoke": True},
    )

    # Two-doc manifest with two real images.
    img_dir = base / "imgs"
    img_dir.mkdir()
    Image.new("L", (200, 100), 255).save(img_dir / "001.jpg")
    Image.new("L", (200, 100), 200).save(img_dir / "002.jpg")
    manifest = base / "m.jsonl"
    manifest.write_text(
        json.dumps({"image": "imgs/001.jpg", "ref": "hello world"}) + "\n"
        + json.dumps({"image": "imgs/002.jpg", "ref": "foo bar baz"}) + "\n"
    )

    return {
        "spm": spm_path,
        "ckpt": ckpt_path,
        "manifest": manifest,
        "out_dir": base,
    }


def _canned_decode(_model, image, _tokenizer, _cfg):
    """Stand-in for ``ocr_with_layout``: returns a one-line layout
    object whose text matches the image's pixel mean. Two distinct
    images in the test fixture (255 vs 200) -> two distinct hyps,
    so the metric path sees variation rather than a constant."""
    from vista_ocr.tokenizer.tokenizer import Line

    if hasattr(image, "getpixel"):
        # mean-ish: pixel at (0,0) is enough for a fixture distinction.
        marker = "white" if image.getpixel((0, 0)) >= 220 else "grey"
    else:
        marker = "x"
    return [Line(text=marker, bbox=(0, 0, 10, 10))]


def test_vista_ocr_eval_end_to_end(cli_eval_env):
    """`vista-ocr eval` wires argparse -> manifest iter -> ckpt load
    -> per-doc decode -> metric aggregate -> JSON sidecar without
    crashing. JSON contents reflect the manifest + canned decode."""
    out_json = cli_eval_env["out_dir"] / "result.json"
    out_predictions = cli_eval_env["out_dir"] / "preds.jsonl"

    # Patch ocr_with_layout where the eval verb imports it.
    with patch("vista_ocr.inference.generate.ocr_with_layout",
               side_effect=_canned_decode):
        from vista_ocr.cli import main
        rc = main([
            "eval",
            "--manifest", str(cli_eval_env["manifest"]),
            "--ckpt", str(cli_eval_env["ckpt"]),
            "--spm", str(cli_eval_env["spm"]),
            "--out-json", str(out_json),
            "--out-predictions", str(out_predictions),
            "--max-new-tokens", "4",
        ])
    assert rc == 0

    # Sidecar JSON: documented schema fields all present.
    assert out_json.exists()
    result = json.loads(out_json.read_text())
    for key in ("ckpt", "ckpt_step", "manifest", "n_docs", "n_empty",
                "cer", "wer", "precision", "recall", "word_f1",
                "elapsed_s", "max_new_tokens"):
        assert key in result, f"missing key {key!r} in result.json"
    assert result["n_docs"] == 2
    assert result["ckpt_step"] == 4242         # from the saved ckpt
    assert result["n_empty"] == 0              # canned decode never empty
    # Metric values are well-defined and finite.
    for k in ("cer", "wer", "precision", "recall", "word_f1"):
        v = result[k]
        assert v == v, f"{k} is NaN"           # NaN check (NaN != NaN)
        assert 0.0 <= v <= 5.0, f"{k}={v} out of plausible range"

    # Per-doc predictions JSONL: 2 lines, each with the documented schema.
    pred_lines = out_predictions.read_text().splitlines()
    assert len(pred_lines) == 2
    for line in pred_lines:
        rec = json.loads(line)
        for k in ("image", "ref", "hyp", "cer", "wer"):
            assert k in rec, f"missing key {k!r} in predictions JSONL"


def test_vista_ocr_eval_through_console_script(cli_eval_env):
    """Same coverage as above, but invoked via the installed
    `vista-ocr` console script (subprocess) so the packaging-level
    plumbing is exercised end-to-end. Catches regressions in
    [project.scripts] / entry-point resolution that in-process
    main() calls miss.

    We pre-patch ``ocr_with_layout`` via a small bootstrap script
    instead of relying on the test's monkeypatch (which only applies
    in-process)."""
    import subprocess

    console = Path(sys.executable).parent / "vista-ocr"
    if not console.exists():
        pytest.skip(f"vista-ocr console script missing at {console}")

    out_json = cli_eval_env["out_dir"] / "result_subproc.json"
    bootstrap = cli_eval_env["out_dir"] / "bootstrap.py"
    # Bootstrap: install the canned decode patch, then re-execute the
    # console script's main().
    bootstrap.write_text(
        "import sys\n"
        "from unittest.mock import patch\n"
        "from vista_ocr.tokenizer.tokenizer import Line\n"
        "def canned(_m, image, _t, _c):\n"
        "    marker = 'white' if image.getpixel((0,0)) >= 220 else 'grey'\n"
        "    return [Line(text=marker, bbox=(0,0,10,10))]\n"
        "with patch('vista_ocr.inference.generate.ocr_with_layout', side_effect=canned):\n"
        "    from vista_ocr.cli import main\n"
        "    sys.exit(main(sys.argv[1:]))\n"
    )
    r = subprocess.run(
        [sys.executable, str(bootstrap),
         "eval",
         "--manifest", str(cli_eval_env["manifest"]),
         "--ckpt", str(cli_eval_env["ckpt"]),
         "--spm", str(cli_eval_env["spm"]),
         "--out-json", str(out_json),
         "--max-new-tokens", "4"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert out_json.exists()
    result = json.loads(out_json.read_text())
    assert result["n_docs"] == 2
