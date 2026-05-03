"""Behavioural-equivalence tests: ``scripts/stage{1,2,3}_run.py`` vs
``vista_ocr.entrypoints.stage{1,2,3}``.

Three contracts per stage:

1. **Flag-set equivalence** -- the legacy script and the entrypoint
   declare the same set of long-form option strings (``--foo``).
   Catches "operator added a flag to scripts/ and forgot the
   entrypoint" drift.
2. **TrainConfig snapshot equivalence** -- under a fixed minimal
   arglist with ``train()`` monkeypatched to capture its ``cfg``,
   the resulting :class:`TrainConfig` dataclasses are equal field-
   for-field. Catches drift in *defaults* that the flag-set test
   misses (e.g., ``warmup_steps``, ``lambda_text``, ``label_smoothing``
   are hardcoded inside main()).
3. **Model state_dict() key-set equivalence** -- the constructed
   model has the same parameter keys (architecture). Catches drift
   in encoder / decoder construction that lives outside TrainConfig
   (e.g., ``n_layers``, ``ffn_dim``).

Both paths are driven through ``runpy.run_path`` (legacy script) /
direct ``main()`` (entrypoint) with ``train()`` and the dataloader
monkeypatched out so we capture state without actually running
training. ``--init-decoder-from random`` keeps the test offline (no
HF Donut download).
"""
from __future__ import annotations

import argparse
import dataclasses
import runpy
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT_PATHS = {
    "stage1": REPO / "scripts" / "stage1_run.py",
    "stage2": REPO / "scripts" / "stage2_run.py",
    "stage3": REPO / "scripts" / "stage3_run.py",
}
ENTRY_MODULES = {
    "stage1": "vista_ocr.entrypoints.stage1",
    "stage2": "vista_ocr.entrypoints.stage2",
    "stage3": "vista_ocr.entrypoints.stage3",
}


# ---------------- Test fixtures (shared across the three stages) ----------


@pytest.fixture(scope="module")
def equiv_env(tmp_path_factory):
    """A minimal training environment: real SPM model, two empty PDFA
    tar shards, an out dir. Same fixtures used by both equivalence
    paths so we can capture state without ever opening real data."""
    from vista_ocr.tokenizer.build_spm import train_spm
    from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
    from vista_ocr.tokenizer.tokenizer import list_special_and_spatial_tokens

    base = tmp_path_factory.mktemp("equiv")
    spm_dir = base / "spm"
    spm_dir.mkdir()
    corpus = spm_dir / "c.txt"
    corpus.write_text(
        (
            "hello world FOO BAR baz qux quux corge\n"
            "abc def ghi jkl mno pqr stu vwx yz\n"
            "the quick brown fox jumps over the lazy dog\n"
            "Sphinx of black quartz judge my vow\n"
        ) * 200,
        encoding="utf-8",
    )
    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    # The scripts/entrypoints hard-code the production grid (3508x2480
    # canvas, 10-px quantizer); that's ~640 spatial tokens. Vocab must
    # leave headroom for them + the BPE pieces.
    train_spm(corpus, spm_dir / "tr", vocab_size=800,
              user_symbols=list_special_and_spatial_tokens(grid))
    spm_path = spm_dir / "tr.model"

    # Two empty PDFA tar shards. They never get opened because the
    # equivalence tests monkeypatch out the dataloader.
    train_shard = base / "pdfa-eng-train-0042.tar"
    val_shard = base / "pdfa-eng-train-0043.tar"
    for p in (train_shard, val_shard):
        with tarfile.open(p, "w") as tf:
            pass

    return {
        "spm": spm_path,
        "train_shard": train_shard,
        "val_shard": val_shard,
        "out": base,
    }


def _common_args(env: dict, stage: str) -> list[str]:
    """The minimal arglist that satisfies every required flag of the
    stage scripts; everything else takes its default."""
    out = env["out"] / stage
    out.mkdir(exist_ok=True)
    args = [
        "--train-shards", str(env["train_shard"]),
        "--val-shard", str(env["val_shard"]),
        "--spm", str(env["spm"]),
        "--out", str(out),
        "--steps", "1",          # max_steps capped; train() is mocked anyway
        "--page-h", "128", "--page-w", "128",
    ]
    if stage == "stage1":
        # --init-decoder-from random keeps the test offline.
        args += ["--init-decoder-from", "random"]
    return args


def _capture_via_module(module_path: str, argv: list[str]) -> dict:
    """Run ``vista_ocr.entrypoints.stageN.main(argv)`` with ``train()``
    and the dataloader / dataset iters mocked out. Returns
    ``{model, tokenizer, cfg, model_state_keys}``."""
    import importlib

    captured: dict = {}

    def fake_train(model=None, sample_stream=None, tokenizer=None, cfg=None,
                   *, max_steps=None, on_step=None, **_kw):
        # Accept both ``train(model, sample_stream=..., ...)`` (the
        # stage 2/3 calling convention) and ``train(model=..., ...)``.
        captured["model"] = model
        captured["tokenizer"] = tokenizer
        captured["cfg"] = cfg
        return []

    def fake_loader(*args, **kwargs):
        return []

    def fake_iter_pdfa(cfg):
        if False:
            yield None  # generator

    # The model is created on .cuda() for stage 2/3; redirect to CPU.
    import torch
    real_to = torch.nn.Module.to

    def fake_to(self, *args, **kwargs):
        # Drop "cuda" device requests; everything else passes through.
        if args and isinstance(args[0], (str, torch.device)):
            return self
        return real_to(self, *args, **kwargs)

    def fake_cuda(self, *args, **kwargs):
        return self

    mod = importlib.import_module(module_path)
    with patch("vista_ocr.training.train_loop.train", fake_train), \
         patch("vista_ocr.data.dataloader.make_pdfa_dataloader", fake_loader), \
         patch("vista_ocr.data.pdfa.iter_pdfa", fake_iter_pdfa), \
         patch.object(torch.nn.Module, "cuda", fake_cuda), \
         patch.object(torch.nn.Module, "to", fake_to):
        try:
            mod.main(argv)
        except SystemExit:
            # main() may raise SystemExit on warning/error paths; what
            # matters is whether train() was reached.
            pass
    if "cfg" not in captured:
        raise AssertionError(
            f"{module_path}.main never reached train(); "
            "the test fixture didn't drive far enough."
        )
    captured["model_state_keys"] = tuple(sorted(captured["model"].state_dict().keys()))
    return captured


def _capture_via_script(script: Path, argv: list[str]) -> dict:
    """Same as :func:`_capture_via_module` but runs the legacy script
    via ``runpy.run_path`` so its ``if __name__ == '__main__'`` block
    fires."""
    captured: dict = {}

    def fake_train(model=None, sample_stream=None, tokenizer=None, cfg=None,
                   *, max_steps=None, on_step=None, **_kw):
        # Accept both ``train(model, sample_stream=..., ...)`` (the
        # stage 2/3 calling convention) and ``train(model=..., ...)``.
        captured["model"] = model
        captured["tokenizer"] = tokenizer
        captured["cfg"] = cfg
        return []

    def fake_loader(*args, **kwargs):
        return []

    def fake_iter_pdfa(cfg):
        if False:
            yield None

    import torch
    real_to = torch.nn.Module.to

    def fake_to(self, *args, **kwargs):
        if args and isinstance(args[0], (str, torch.device)):
            return self
        return real_to(self, *args, **kwargs)

    def fake_cuda(self, *args, **kwargs):
        return self

    fake_argv = [str(script)] + argv
    with patch.object(sys, "argv", fake_argv), \
         patch("vista_ocr.training.train_loop.train", fake_train), \
         patch("vista_ocr.data.dataloader.make_pdfa_dataloader", fake_loader), \
         patch("vista_ocr.data.pdfa.iter_pdfa", fake_iter_pdfa), \
         patch.object(torch.nn.Module, "cuda", fake_cuda), \
         patch.object(torch.nn.Module, "to", fake_to):
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit:
            pass
    if "cfg" not in captured:
        raise AssertionError(
            f"{script} never reached train(); fixture insufficient."
        )
    captured["model_state_keys"] = tuple(sorted(captured["model"].state_dict().keys()))
    return captured


# ---------------- Equivalence test 1: --help flag-set ---------------------


def _flag_set_from_parser(parser: argparse.ArgumentParser) -> set[str]:
    flags: set[str] = set()
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        for opt in action.option_strings:
            if opt.startswith("--"):
                flags.add(opt)
    return flags


def _flag_set_from_help(help_text: str) -> set[str]:
    import re

    flags = set(re.findall(r"\s(--[\w-]+)", help_text))
    flags.discard("--help")  # built-in argparse action; excluded from comparison
    return flags


@pytest.mark.parametrize("stage", ["stage1", "stage2", "stage3"])
def test_flag_set_equivalence(stage: str):
    """Both the legacy script and the entrypoint declare the same
    long-form ``--flag`` set. Catches forgotten flag mirrors."""
    import importlib

    legacy_help = subprocess.run(
        [sys.executable, str(SCRIPT_PATHS[stage]), "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert legacy_help.returncode == 0, legacy_help.stderr
    legacy_flags = _flag_set_from_help(legacy_help.stdout)

    entry_mod = importlib.import_module(ENTRY_MODULES[stage])
    entry_flags = _flag_set_from_parser(entry_mod.build_parser())

    missing = legacy_flags - entry_flags
    extra = entry_flags - legacy_flags
    assert not missing, (
        f"{stage}: entrypoint missing flags from legacy script: {missing}"
    )
    assert not extra, (
        f"{stage}: entrypoint has extra flags not in legacy script: {extra}"
    )


# ---------------- Equivalence test 2: TrainConfig snapshot ---------------


def _normalize_cfg(cfg) -> dict:
    """Strip non-comparable fields (closures + paths in checkpoint)
    so the snapshot compares the dataclass values that matter for
    behavioural drift."""
    d = dataclasses.asdict(cfg)
    # val_batches_factory and val_loss_fn / val_decode_fn are closures
    # over tokenizer/spm path; we can't compare them by identity.
    for k in ("val_batches_factory", "val_loss_fn", "val_decode_fn"):
        d.pop(k, None)
    return d


@pytest.mark.parametrize("stage", ["stage1", "stage2", "stage3"])
def test_trainconfig_snapshot_equivalence(stage: str, equiv_env):
    argv = _common_args(equiv_env, stage)
    legacy = _capture_via_script(SCRIPT_PATHS[stage], argv)
    new = _capture_via_module(ENTRY_MODULES[stage], argv)

    legacy_cfg = _normalize_cfg(legacy["cfg"])
    new_cfg = _normalize_cfg(new["cfg"])

    diffs = {
        k: (legacy_cfg[k], new_cfg[k])
        for k in legacy_cfg
        if legacy_cfg[k] != new_cfg.get(k)
    }
    assert not diffs, f"{stage} TrainConfig drift: {diffs}"


# ---------------- Equivalence test 3: model state_dict() keys ------------


@pytest.mark.parametrize("stage", ["stage1", "stage2", "stage3"])
def test_model_state_dict_keys_equivalence(stage: str, equiv_env):
    argv = _common_args(equiv_env, stage)
    legacy = _capture_via_script(SCRIPT_PATHS[stage], argv)
    new = _capture_via_module(ENTRY_MODULES[stage], argv)
    assert legacy["model_state_keys"] == new["model_state_keys"], (
        f"{stage}: model architecture drift between scripts/ and entrypoints/"
    )
