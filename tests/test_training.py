"""Training-loop and schedule tests.

The overfit-on-one-image test is the smoke gate for task #13: it verifies
that gradient flow, tokenizer, loss and model wiring are all correct on
CPU before we touch the GPU VM.
"""
from __future__ import annotations

from itertools import cycle, islice

import pytest
import torch

from vista_ocr.data.synth.synthdog_bbox import SynthDogConfig, generate_sample
from vista_ocr.models.decoder import small_random_decoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.build_spm import train_spm
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import (
    VistaTokenizer,
    list_special_and_spatial_tokens,
)
from vista_ocr.training.schedules import exponential_dropout, linear_warmup_cosine
from vista_ocr.training.train_loop import TrainConfig, train


def test_linear_warmup_cosine_shape():
    cfg = {"warmup_steps": 10, "total_steps": 100, "base_lr": 1e-3}
    assert linear_warmup_cosine(0, **cfg) < 1e-3
    assert linear_warmup_cosine(9, **cfg) <= 1e-3
    assert linear_warmup_cosine(10, **cfg) == pytest.approx(1e-3)
    # After total_steps it stays at the floor.
    assert linear_warmup_cosine(200, **cfg) == 0.0


def test_linear_warmup_cosine_min_lr_ratio_floor():
    """A4: with min_lr_ratio=0.05 the schedule must floor at 5% of base."""
    cfg = {"warmup_steps": 10, "total_steps": 100, "base_lr": 1e-3,
           "min_lr_ratio": 0.05}
    # at total_steps and beyond
    assert linear_warmup_cosine(100, **cfg) == pytest.approx(5e-5)
    assert linear_warmup_cosine(500, **cfg) == pytest.approx(5e-5)
    # mid-decay strictly above the floor
    assert linear_warmup_cosine(50, **cfg) > 5e-5


def test_linear_warmup_cosine_handoff_is_continuous():
    """A4 gap: the warmup -> cosine handoff must not jump.

    At step == warmup_steps - 1 (last warmup) and step == warmup_steps
    (first cosine) the LR must be ~base_lr; the difference must be
    negligible. A bug here would be invisible to the floor test.
    """
    cfg = {"warmup_steps": 100, "total_steps": 1000, "base_lr": 1e-3,
           "min_lr_ratio": 0.05}
    last_warmup = linear_warmup_cosine(99, **cfg)
    first_cosine = linear_warmup_cosine(100, **cfg)
    assert last_warmup == pytest.approx(1e-3, rel=1e-9)
    assert first_cosine == pytest.approx(1e-3, rel=1e-9)
    # And the slope across the handoff is finite (no spike).
    next_cosine = linear_warmup_cosine(101, **cfg)
    assert abs(first_cosine - next_cosine) < 1e-3 * 1e-3  # tiny step


def test_exponential_dropout_grows_to_one():
    assert exponential_dropout(0) == pytest.approx(0.0)
    assert exponential_dropout(int(1e6)) > 0.999


@pytest.fixture(scope="module")
def grid() -> SpatialGrid:
    return SpatialGrid(canvas_h=128, canvas_w=128, quantizer_px=4, scheme="original")


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory, grid):
    tmp = tmp_path_factory.mktemp("spm_tr")
    corpus = tmp / "c.txt"
    corpus.write_text(
        (
            "hello world\nfoo bar\nthe quick brown fox jumps over the lazy dog\n"
            "abc def ghi jkl mno pqr stu vwx yz\n"
            "Sphinx of black quartz judge my vow\n"
            "Pack my box with five dozen liquor jugs\n"
            "0 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100\n"
        ) * 400,
        encoding="utf-8",
    )
    out = tmp / "tr"
    train_spm(corpus, out, vocab_size=180, user_symbols=list_special_and_spatial_tokens(grid))
    return VistaTokenizer(out.with_suffix(".model"), grid)


@pytest.fixture(scope="module")
def tiny_model(tokenizer: VistaTokenizer) -> VistaOCR:
    torch.manual_seed(0)
    enc = FCNEncoderWidther(input_channels=1, dropout=0.0)
    dec = small_random_decoder(
        vocab_size=tokenizer.vocab_size,
        d_model=1024,
        n_layers=1,
        n_heads=4,
        ffn_dim=128,
    )
    return VistaOCR(encoder=enc, decoder=dec)


def test_grad_accum_gates_optimizer_step(
    tiny_model: VistaOCR, tokenizer: VistaTokenizer,
):
    """Phase 2: with grad_accum_steps=N, parameters update once every
    N forward passes. Catches a regression where accum collapses to 1."""
    import copy
    sample = generate_sample(["hi"], SynthDogConfig(canvas_h=64, canvas_w=64, line_height=20))
    cfg = TrainConfig(
        base_lr=1e-3, warmup_steps=0, total_steps=12,
        micro_batch_size=1, grad_accum_steps=4,
        target_h=128, target_w=128, pad_multiple=32,
    )
    # Two clones so we measure the same starting state independently.
    model_3 = copy.deepcopy(tiny_model)
    model_4 = copy.deepcopy(tiny_model)
    init_params = next(tiny_model.parameters()).detach().clone()

    # 3 forwards with accum=4 -> not enough for an optimizer step.
    train(model_3, [sample] * 3, tokenizer, cfg, max_steps=3)
    after_three = next(model_3.parameters()).detach()
    assert torch.equal(init_params, after_three), (
        "Parameters changed before grad_accum_steps were accumulated"
    )

    # 4 forwards with accum=4 -> exactly one optimizer step.
    train(model_4, [sample] * 4, tokenizer, cfg, max_steps=4)
    after_four = next(model_4.parameters()).detach()
    assert not torch.equal(init_params, after_four), (
        "Parameters did not update after the full accumulation window"
    )


def test_save_final_writes_ckpt_final_at_end_of_training(
    tiny_model: VistaOCR, tokenizer: VistaTokenizer, tmp_path,
):
    """Phase 1.6: with save_final=True (default), ckpt_final.pt is
    written at the end of training in addition to any ckpt_best.pt /
    periodic ckpts. The final ckpt captures the last-step state, not
    val-loss-selected; downstream evals may want both."""
    from vista_ocr.training.callbacks import CheckpointConfig

    sample = generate_sample(["hi"], SynthDogConfig(canvas_h=64, canvas_w=64, line_height=20))
    cfg = TrainConfig(
        base_lr=1e-3, warmup_steps=1, total_steps=4,
        micro_batch_size=1, grad_accum_steps=1,
        target_h=128, target_w=128, pad_multiple=32,
        checkpoint=CheckpointConfig(out_dir=tmp_path, save_every=999, keep_last=3),
    )
    train(tiny_model, [sample] * 3, tokenizer, cfg, max_steps=3)
    assert (tmp_path / "ckpt_final.pt").exists()


def test_save_final_can_be_disabled(
    tiny_model: VistaOCR, tokenizer: VistaTokenizer, tmp_path,
):
    from vista_ocr.training.callbacks import CheckpointConfig

    sample = generate_sample(["hi"], SynthDogConfig(canvas_h=64, canvas_w=64, line_height=20))
    cfg = TrainConfig(
        base_lr=1e-3, warmup_steps=1, total_steps=4,
        micro_batch_size=1, grad_accum_steps=1,
        target_h=128, target_w=128, pad_multiple=32,
        checkpoint=CheckpointConfig(
            out_dir=tmp_path, save_every=999, keep_last=3, save_final=False,
        ),
    )
    train(tiny_model, [sample] * 3, tokenizer, cfg, max_steps=3)
    assert not (tmp_path / "ckpt_final.pt").exists()


def test_compile_fallback_keeps_training_running(
    tiny_model: VistaOCR, tokenizer: VistaTokenizer, monkeypatch, caplog,
):
    """Phase 6: when ``compile_model=True`` but ``torch.compile`` raises,
    the train loop logs a WARNING and continues with the eager model.
    Catches the regression where a Compile failure would stop a multi-
    day run at hour 0."""
    import copy
    import logging
    import torch as _torch

    def _failing_compile(model, **_kwargs):
        raise RuntimeError("simulated compile failure")

    monkeypatch.setattr(_torch, "compile", _failing_compile)

    # Use a deepcopy so the module-scoped fixture isn't mutated -- the
    # overfit test downstream expects the fresh init.
    model = copy.deepcopy(tiny_model)
    sample = generate_sample(["hi"], SynthDogConfig(canvas_h=64, canvas_w=64, line_height=20))
    cfg = TrainConfig(
        base_lr=1e-3, warmup_steps=0, total_steps=2,
        micro_batch_size=1, grad_accum_steps=1,
        target_h=128, target_w=128, pad_multiple=32,
        compile_model=True,
    )
    with caplog.at_level(logging.WARNING):
        history = train(model, [sample], tokenizer, cfg, max_steps=1)
    assert len(history) == 1
    assert any("compile failed" in r.message.lower() for r in caplog.records)


def test_one_optim_step_runs(tiny_model: VistaOCR, tokenizer: VistaTokenizer):
    sample = generate_sample(["hi"], SynthDogConfig(canvas_h=64, canvas_w=64, line_height=20))
    cfg = TrainConfig(
        base_lr=1e-3,
        warmup_steps=1,
        total_steps=4,
        micro_batch_size=1,
        grad_accum_steps=1,
        target_h=128,
        target_w=128,
        pad_multiple=32,
    )
    history = train(tiny_model, [sample], tokenizer, cfg, max_steps=1)
    assert len(history) == 1
    assert history[0].lr > 0


def test_overfit_single_sample_loss_decreases(
    tiny_model: VistaOCR, tokenizer: VistaTokenizer
):
    """Smoke test (task #13). On a 1-sample dataset, the loss should drop
    monotonically over a few dozen steps."""
    sample = generate_sample(
        ["abc"], SynthDogConfig(canvas_h=64, canvas_w=64, line_height=24)
    )
    sample.task = "ocr"

    cfg = TrainConfig(
        base_lr=3e-3,
        warmup_steps=2,
        total_steps=60,
        micro_batch_size=1,
        grad_accum_steps=1,
        target_h=128,
        target_w=128,
        pad_multiple=32,
        lambda_text=1.0,             # OCR-only sample has no spatial tokens.
    )
    stream = cycle([sample])
    history = train(tiny_model, islice(stream, 60), tokenizer, cfg, max_steps=60)

    first_loss = sum(h.loss for h in history[:3]) / 3
    last_loss = sum(h.loss for h in history[-3:]) / 3
    assert last_loss < first_loss * 0.6, (first_loss, last_loss)


def test_dropout_schedule_off_by_default(tiny_model: VistaOCR, tokenizer: VistaTokenizer):
    """B1: encoder_dropout_max=None must NOT mutate encoder dropout."""
    from vista_ocr.models.encoder import _MixDropout
    # baseline dropout values across all MixDropout modules
    pre = [m.dropout.p for m in tiny_model.encoder.modules() if isinstance(m, _MixDropout)]
    sample = generate_sample(["hi"], SynthDogConfig(canvas_h=64, canvas_w=64))
    sample.task = "ocr"
    cfg = TrainConfig(
        base_lr=1e-4, warmup_steps=1, total_steps=4,
        target_h=128, target_w=128, pad_multiple=32,
        lambda_text=1.0,
        encoder_dropout_max=None,             # disabled
    )
    train(tiny_model, [sample], tokenizer, cfg, max_steps=1)
    post = [m.dropout.p for m in tiny_model.encoder.modules() if isinstance(m, _MixDropout)]
    assert pre == post


def test_dropout_schedule_rises_when_enabled(tiny_model: VistaOCR, tokenizer: VistaTokenizer):
    """B1: with encoder_dropout_max=0.5 and dropout_T=2, p should grow
    quickly across a handful of steps (1 - exp(-step/2))."""
    from vista_ocr.models.encoder import _MixDropout
    sample = generate_sample(["hi"], SynthDogConfig(canvas_h=64, canvas_w=64))
    sample.task = "ocr"
    cfg = TrainConfig(
        base_lr=1e-4, warmup_steps=1, total_steps=10,
        target_h=128, target_w=128, pad_multiple=32,
        lambda_text=1.0,
        encoder_dropout_max=0.5, dropout_T=2.0,
    )
    train(tiny_model, list(__import__("itertools").repeat(sample, 5)),
          tokenizer, cfg, max_steps=5)
    final = next(m.dropout.p for m in tiny_model.encoder.modules()
                 if isinstance(m, _MixDropout))
    # at step 5 with T=2, p = 0.5 * (1 - exp(-5/2)) ≈ 0.459
    assert 0.35 < final < 0.55, final


def test_run_validation_decode_n_zero_skips_decode():
    """B3: decode_n=0 keeps the no-decode path bit-exact."""
    import torch
    from torch import nn

    from vista_ocr.training.callbacks import run_validation
    m = nn.Linear(2, 1)

    def loss_fn(model, batch):
        return torch.tensor(0.5)

    out = run_validation(m, iter([0, 1, 2]), loss_fn, max_batches=2, decode_n=0)
    assert "val_cer" not in out
    assert out["n_batches"] == 2


def test_run_validation_decode_n_emits_metrics():
    """B3: with decode_n>0 we get cer/wer/word-f1 in the result."""
    import torch
    from torch import nn

    from vista_ocr.training.callbacks import run_validation
    m = nn.Linear(2, 1)
    refs_batches = [(0, "hello world"), (1, "foo bar")]

    def loss_fn(model, batch):
        return torch.tensor(0.1)

    def decode_fn(model, batch):
        idx, gt = batch
        return [gt], [gt]   # perfect predictions

    out = run_validation(
        m, iter(refs_batches), loss_fn, max_batches=2,
        decode_fn=decode_fn, decode_n=2,
    )
    assert out["val_cer"] == pytest.approx(0.0)
    assert out["val_wer"] == pytest.approx(0.0)
    assert out["val_word_f1"] == pytest.approx(1.0)
    assert out["val_decoded_n"] == 2
    assert out["val_decoded_n_empty"] == 0


def test_run_validation_counts_empty_hypotheses():
    """B3 gap: empty decoder outputs (silent failures) must show up in
    val_decoded_n_empty so an operator notices a model collapsing to
    EOS-only generations."""
    import torch
    from torch import nn

    from vista_ocr.training.callbacks import run_validation
    m = nn.Linear(2, 1)

    def loss_fn(model, batch):
        return torch.tensor(0.1)

    def decode_fn(model, batch):
        idx, gt = batch
        # First batch: model emits empty string (EOS-only collapse).
        # Second batch: model emits a real prediction.
        if idx == 0:
            return [gt], [""]
        return [gt], [gt]

    out = run_validation(
        m, iter([(0, "hello world"), (1, "foo bar")]),
        loss_fn, max_batches=2, decode_fn=decode_fn, decode_n=2,
    )
    assert out["val_decoded_n"] == 2
    assert out["val_decoded_n_empty"] == 1


def test_run_validation_warns_on_majority_empty(caplog):
    """B3: empty-output sanity check fires when most decodes are empty."""
    import logging as _logging

    import torch
    from torch import nn

    from vista_ocr.training.callbacks import run_validation
    m = nn.Linear(2, 1)

    def loss_fn(model, batch):
        return torch.tensor(0.1)

    def decode_fn(model, batch):
        return ["target"], [""]   # always empty prediction

    with caplog.at_level(_logging.WARNING, logger="vista_ocr.training.callbacks"):
        out = run_validation(
            m, iter(range(3)), loss_fn, max_batches=3,
            decode_fn=decode_fn, decode_n=3,
        )
    assert out["val_decoded_n_empty"] == 3
    assert any("empty" in rec.message for rec in caplog.records)


def test_optimizer_excludes_frozen_params(tiny_model: VistaOCR):
    from vista_ocr.training.train_loop import make_optimizer
    tiny_model.freeze_decoder(True)
    cfg = TrainConfig(base_lr=1e-4)
    opt = make_optimizer(tiny_model, cfg)
    opt_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    decoder_param_ids = {id(p) for p in tiny_model.decoder.parameters()}
    assert decoder_param_ids.isdisjoint(opt_param_ids)
    tiny_model.freeze_decoder(False)


def test_optimizer_uses_mbart_betas_and_eps(tiny_model: VistaOCR):
    from vista_ocr.training.train_loop import make_optimizer
    cfg = TrainConfig(adam_betas=(0.9, 0.98), adam_eps=1e-6)
    opt = make_optimizer(tiny_model, cfg)
    for g in opt.param_groups:
        assert g["betas"] == (0.9, 0.98)
        assert g["eps"] == 1e-6


def test_freeze_decoder_disables_decoder_grads(
    tiny_model: VistaOCR, tokenizer: VistaTokenizer
):
    sample = generate_sample(["hi"], SynthDogConfig(canvas_h=64, canvas_w=64))
    sample.task = "ocr"
    cfg = TrainConfig(
        base_lr=1e-4,
        warmup_steps=1,
        total_steps=4,
        target_h=128,
        target_w=128,
        pad_multiple=32,
        freeze_decoder=True,
        lambda_text=1.0,
    )
    train(tiny_model, [sample], tokenizer, cfg, max_steps=1)
    assert all(not p.requires_grad for p in tiny_model.decoder.parameters())
    # Restore.
    tiny_model.freeze_decoder(False)


# ---------- D2: early-stop train_loop integration ---------------------------
#
# The 11 unit tests in tests/test_early_stop.py cover ``early_stop_decision``
# in isolation. These three tests cover the wiring through ``train()``:
# (1) a flat val curve aborts the loop and emits the structured
# ``EARLY_STOP:`` log line; (2) the early-stop state is persisted into
# ``ckpt_final.pt`` so a forensic reader can see why the run stopped;
# (3) ``resume_from`` restores the prior counters so a kill+restart does
# not pay another full ``patience * val_every`` window before aborting.

def _flat_val_train_cfg(out_dir, *, warmup_vals, patience, val_every=1):
    """Build a TrainConfig that fires val every step with a constant
    val_loss, so the early-stop EMA never improves."""
    from vista_ocr.training.callbacks import (
        CheckpointConfig,
        EarlyStopConfig,
        ValConfig,
    )

    def loss_fn(_model, _batch):
        return torch.tensor(1.234)

    return TrainConfig(
        base_lr=1e-3, warmup_steps=0, total_steps=200,
        micro_batch_size=1, grad_accum_steps=1,
        target_h=128, target_w=128, pad_multiple=32,
        checkpoint=CheckpointConfig(
            out_dir=out_dir, save_every=999, keep_last=3,
        ),
        val=ValConfig(every=val_every, max_batches=1),
        val_batches_factory=lambda: iter([0]),
        val_loss_fn=loss_fn,
        early_stop=EarlyStopConfig(
            enabled=True,
            patience=patience,
            min_delta=0.01,
            smooth_window=3,
            warmup_vals=warmup_vals,
            spike_threshold=10.0,    # silence spike branch
            spike_consecutive=999,
        ),
    )


def test_early_stop_aborts_train_loop_with_structured_log(
    tiny_model: VistaOCR, tokenizer: VistaTokenizer, tmp_path, caplog,
):
    """D2: with a flat val curve and patience=2 / warmup_vals=1 the
    train loop must return before ``max_steps`` and emit the structured
    ``EARLY_STOP:`` line that the side-process tailer parses."""
    import copy
    import logging

    model = copy.deepcopy(tiny_model)
    sample = generate_sample(["hi"], SynthDogConfig(canvas_h=64, canvas_w=64, line_height=20))
    cfg = _flat_val_train_cfg(tmp_path, warmup_vals=1, patience=2)

    with caplog.at_level(logging.INFO, logger="vista_ocr.training.train_loop"):
        # Plenty of samples so micro_iter doesn't run dry before the
        # abort decision fires; max_steps caps the upper bound.
        history = train(
            model, list(islice(cycle([sample]), 100)),
            tokenizer, cfg, max_steps=50,
        )

    # First val sets smoothed_best (counter=0). val 2 is warmup.
    # vals 3, 4 increment no_improve to 2 == patience -> abort at step 4.
    assert len(history) <= 10, (
        f"train_loop should have early-stopped well before max_steps; "
        f"saw {len(history)} steps"
    )
    structured = [r for r in caplog.records if r.message.startswith("EARLY_STOP: ")]
    assert structured, "Expected a structured EARLY_STOP: log line"
    msg = structured[0].message
    for needle in ("step=", "reason=patience_exceeded",
                   "smoothed_best=", "patience=2",
                   "no_improve=", "spike="):
        assert needle in msg, f"missing field {needle!r} in {msg!r}"


def test_early_stop_state_persists_to_checkpoint(
    tiny_model: VistaOCR, tokenizer: VistaTokenizer, tmp_path,
):
    """D2: after early-stop fires, ``ckpt_final.pt`` carries
    ``reason=early_stop`` plus a round-trippable ``early_stop_state``
    dict. Forensic readers (and the resume path in the next test)
    depend on this."""
    import copy

    from vista_ocr.training.callbacks import EarlyStopState, load_checkpoint

    model = copy.deepcopy(tiny_model)
    sample = generate_sample(["hi"], SynthDogConfig(canvas_h=64, canvas_w=64, line_height=20))
    cfg = _flat_val_train_cfg(tmp_path, warmup_vals=1, patience=2)

    train(model, list(islice(cycle([sample]), 100)),
          tokenizer, cfg, max_steps=50)

    final = tmp_path / "ckpt_final.pt"
    assert final.exists(), "early-stop must write ckpt_final.pt"

    # Use load_checkpoint so this also exercises the resume code path.
    from vista_ocr.training.train_loop import make_optimizer
    opt = make_optimizer(model, cfg)
    payload = load_checkpoint(final, model=model, optimizer=opt)
    assert payload.extra.get("reason") == "early_stop"
    assert payload.extra.get("early_stop_reason") == "patience_exceeded"

    state = EarlyStopState.from_dict(payload.extra.get("early_stop_state"))
    assert state.n_vals_seen >= 3
    assert state.no_improve_counter >= cfg.early_stop.patience
    # smoothed_best converged to the constant val_loss.
    assert abs(state.smoothed_best - 1.234) < 1e-3


def test_early_stop_state_restores_on_resume(
    tiny_model: VistaOCR, tokenizer: VistaTokenizer, tmp_path, caplog,
):
    """D2 / F2: a checkpoint whose ``early_stop_state`` already has
    ``no_improve_counter == patience - 1`` must abort on the very next
    val pass. Without state restore the resumed run would reset the
    counter and pay another ``patience * val_every`` window before
    aborting -- which is exactly the regression F2 was meant to
    prevent."""
    import copy
    import logging

    from vista_ocr.training.callbacks import (
        EarlyStopState,
        save_checkpoint,
    )
    from vista_ocr.training.train_loop import make_optimizer

    model = copy.deepcopy(tiny_model)
    sample = generate_sample(["hi"], SynthDogConfig(canvas_h=64, canvas_w=64, line_height=20))
    cfg = _flat_val_train_cfg(tmp_path, warmup_vals=0, patience=3)

    # Pre-craft a checkpoint with the patience counter one short of abort.
    crafted = EarlyStopState(
        val_history=[1.234, 1.234, 1.234, 1.234],
        smoothed_best=1.234,
        no_improve_counter=cfg.early_stop.patience - 1,
        spike_counter=0,
        n_vals_seen=4,    # already past warmup_vals=0
    )
    seed_path = tmp_path / "ckpt_seed.pt"
    seed_opt = make_optimizer(model, cfg)
    save_checkpoint(
        seed_path,
        step=0, model=model, optimizer=seed_opt,
        best_val_loss=1.234,
        extra={"early_stop_state": crafted.to_dict()},
    )

    # Resume from the seed; abort should fire on the very first val
    # pass after step 0 (val_every=1, so that's step 1).
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    cfg_resume = _flat_val_train_cfg(resume_dir, warmup_vals=0, patience=3)
    cfg_resume = TrainConfig(
        **{**cfg_resume.__dict__, "resume_from": seed_path},
    )

    with caplog.at_level(logging.INFO, logger="vista_ocr.training.train_loop"):
        history = train(
            model, list(islice(cycle([sample]), 50)),
            tokenizer, cfg_resume, max_steps=10,
        )

    # Without state restore, abort would only fire at val
    # warmup_vals + patience = 3 (steps 1,2,3 -> abort at step 3+).
    # With state restore, the prior counter (patience-1) plus this
    # val (no improvement on a flat curve) hits patience immediately.
    assert len(history) <= 2, (
        f"resumed run should abort on the first val pass; saw {len(history)}"
    )
    assert any("EARLY_STOP: " in r.message for r in caplog.records)
    assert any("Resumed from" in r.message for r in caplog.records)


# ---------- DS-fix Phase 2: ckpt_best second-pass eval ---------------------
#
# val_decode_n stays small (5) so the per-val log line is cheap. When a
# val pass marks the ckpt as a ckpt_best candidate (val_loss improved),
# train() fires a second eval pass with val_decode_n_best samples and
# persists the larger-sample CER / WER / word-F1 in the checkpoint's
# ``extra["best_candidate"]`` dict. These two tests cover the wiring:
# (1) the second pass fires only on val_loss improvement and lands in
# the ckpt; (2) without val_decode_n_best the prior behaviour is
# preserved bit-exact.

def _candidate_cfg(out_dir, *, decode_n_best, decreasing_loss=True):
    """A train cfg with val every step, a fake decode_fn that returns
    perfect predictions, and a loss_fn whose value drops on each call
    (so every val pass is a ckpt_best candidate)."""
    from vista_ocr.training.callbacks import CheckpointConfig, ValConfig

    # Constant small loss so every val pass beats the prior best.
    # The second eval pass also calls loss_fn (once per batch up to
    # val_decode_n_best), so a fixed-length iterator would exhaust.
    def loss_fn(_model, _batch):
        return torch.tensor(0.1 if decreasing_loss else 0.5)

    refs = ["hello world", "foo bar", "the quick brown fox"] * 200

    def factory():
        return iter([(i, refs[i % len(refs)]) for i in range(300)])

    def decode_fn(_model, batch):
        _i, gt = batch
        return [gt], [gt]   # perfect

    return TrainConfig(
        base_lr=1e-3, warmup_steps=0, total_steps=200,
        micro_batch_size=1, grad_accum_steps=1,
        target_h=128, target_w=128, pad_multiple=32,
        checkpoint=CheckpointConfig(
            out_dir=out_dir, save_every=999, keep_last=3,
        ),
        val=ValConfig(every=1, max_batches=2),
        val_batches_factory=factory,
        val_loss_fn=loss_fn,
        val_decode_fn=decode_fn,
        val_decode_n=2,                  # cheap per-pass decode
        val_decode_n_best=decode_n_best,  # the new knob
    )


def test_ckpt_best_carries_second_pass_metrics_when_decode_n_best_set(
    tiny_model: VistaOCR, tokenizer: VistaTokenizer, tmp_path,
):
    """DS-fix P2: with val_decode_n_best>0, every ckpt_best save fires
    a second eval pass and the larger-sample CER / word_f1 land in
    ckpt['extra']['best_candidate']."""
    import copy

    from vista_ocr.training.callbacks import load_checkpoint
    from vista_ocr.training.train_loop import make_optimizer

    model = copy.deepcopy(tiny_model)
    sample = generate_sample(["hi"], SynthDogConfig(canvas_h=64, canvas_w=64, line_height=20))
    cfg = _candidate_cfg(tmp_path, decode_n_best=64)

    train(model, list(islice(cycle([sample]), 50)),
          tokenizer, cfg, max_steps=3)

    best = tmp_path / "ckpt_best.pt"
    assert best.exists(), "ckpt_best.pt must exist after a val_loss improvement"

    payload = load_checkpoint(
        best, model=model, optimizer=make_optimizer(model, cfg),
    )
    cand = payload.extra.get("best_candidate")
    assert cand is not None, (
        "ckpt_best['extra']['best_candidate'] must carry the second-pass metrics"
    )
    # decode_fn returns perfect predictions, so cer/wer = 0, word_f1 = 1.
    assert cand["val_cer"] == pytest.approx(0.0)
    assert cand["val_word_f1"] == pytest.approx(1.0)
    # The second pass decoded *more* batches than the first (n_best=64
    # vs decode_n=2 per regular val pass).
    assert cand["val_decoded_n"] > cfg.val_decode_n


def test_ckpt_best_no_second_pass_when_decode_n_best_zero(
    tiny_model: VistaOCR, tokenizer: VistaTokenizer, tmp_path,
):
    """DS-fix P2 back-compat: with val_decode_n_best=0 the prior
    behaviour is preserved -- ckpt_best['extra'] does NOT carry a
    'best_candidate' key, and the second pass does not fire."""
    import copy

    from vista_ocr.training.callbacks import load_checkpoint
    from vista_ocr.training.train_loop import make_optimizer

    model = copy.deepcopy(tiny_model)
    sample = generate_sample(["hi"], SynthDogConfig(canvas_h=64, canvas_w=64, line_height=20))
    cfg = _candidate_cfg(tmp_path, decode_n_best=0)

    train(model, list(islice(cycle([sample]), 50)),
          tokenizer, cfg, max_steps=3)

    best = tmp_path / "ckpt_best.pt"
    assert best.exists()
    payload = load_checkpoint(
        best, model=model, optimizer=make_optimizer(model, cfg),
    )
    assert "best_candidate" not in payload.extra
