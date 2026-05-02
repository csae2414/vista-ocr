"""Tests for scripts/log_tailer.py.

Pure unit tests against synthetic log lines. The TensorBoard path is
guarded via ``pytest.importorskip("tensorboard")`` so the test passes
on CPU-only / TB-less boxes.

No real training log content is committed -- every fixture below is
hand-written to mirror the exact format produced by
``vista_ocr.training.train_loop`` and ``scripts/pretrain_supervised.sh``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
TAILER_PATH = REPO / "scripts" / "log_tailer.py"
spec = importlib.util.spec_from_file_location("_log_tailer", TAILER_PATH)
tailer_mod = importlib.util.module_from_spec(spec)
sys.modules["_log_tailer"] = tailer_mod
spec.loader.exec_module(tailer_mod)


def _default_args(**overrides):
    base = dict(
        gate_empty_after_stage=2,
        gate_empty_after_step=5000,
        empty_streak=3,
        stall_min_rate=0.1,
        stall_window=300.0,
        restart_alert_threshold=3,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _ListSink:
    """Drop-in replacement for JsonlSink that collects records in memory."""

    def __init__(self):
        self.records = []

    def write(self, record):
        rec = dict(record)
        rec.pop("ts", None)  # ts is wallclock; not deterministic for tests
        self.records.append(rec)


def _run(lines, args=None, tb=None):
    sink = _ListSink()
    state = tailer_mod.TailerState()
    a = args or _default_args()
    for line in lines:
        tailer_mod.process_line(line, state, sink, tb, a)
    return sink.records, state


# ---------------------------------------------------------------------
# Per-event-type parsing
# ---------------------------------------------------------------------

def test_train_step_parsed():
    lines = ["2026-05-02 07:14:09 | INFO | step=1234 loss=5.50 text=4.10 loc=2.30 lr=2.5e-05"]
    out, _ = _run(lines)
    assert len(out) == 1
    rec = out[0]
    assert rec == {
        "event": "train_step", "attempt": 1, "stage": 0, "step": 1234,
        "loss": 5.5, "text": 4.1, "loc": 2.3, "lr": 2.5e-05,
    }


def test_stage_header_makes_stage_sticky():
    lines = [
        "=== STAGE 2: multimodal pretraining ===",
        "step=0 loss=9.4 text=9.0 loc=9.7 lr=2.5e-08",
    ]
    out, st = _run(lines)
    assert out[0]["event"] == "stage_header"
    assert out[0]["stage"] == 2
    assert out[1]["event"] == "train_step"
    assert out[1]["stage"] == 2
    assert st.stage == 2


def test_val_with_full_decode_metrics():
    lines = [
        "STAGE 2: multimodal",
        "validation step=2000 val_loss=6.39 n=20 (8.8s) cer=1.0 wer=1.0 word-f1=0.0 decoded=5 empty=5",
    ]
    out, _ = _run(lines)
    val = [r for r in out if r["event"] == "val"][0]
    assert val["val_loss"] == 6.39
    assert val["cer"] == 1.0 and val["wer"] == 1.0 and val["word_f1"] == 0.0
    assert val["decoded"] == 5 and val["empty"] == 5


def test_val_loss_only_when_decode_disabled():
    lines = ["validation step=500 val_loss=7.5 n=20 (8.0s)"]
    out, _ = _run(lines)
    val = [r for r in out if r["event"] == "val"][0]
    assert val["val_loss"] == 7.5
    assert "cer" not in val and "decoded" not in val


def test_checkpoint_event():
    lines = ["Checkpoint saved: checkpoints/stage1/ckpt_00010000.pt (step 10000, 609.6 MB)"]
    out, _ = _run(lines)
    assert out == [{"event": "checkpoint", "attempt": 1, "stage": 0, "step": 10000}]


def test_stage_done_summary():
    lines = [
        "STAGE 1: calibration",
        "DONE: 20000 steps, 2889.9s (0.144s/step). Loss first=10.032 last=8.836 delta=1.196",
    ]
    out, _ = _run(lines)
    summ = [r for r in out if r["event"] == "stage_summary"][0]
    assert summ["n_steps"] == 20000
    assert summ["wall_s"] == 2889.9
    assert summ["per_step_s"] == 0.144
    assert summ["loss_first"] == 10.032
    assert summ["loss_last"] == 8.836
    assert summ["loss_delta"] == 1.196


def test_chain_done():
    lines = ["=== 2026-05-02T13:00:00+00:00  ALL STAGES DONE ==="]
    out, _ = _run(lines)
    assert out == [{"event": "chain_done", "attempt": 1}]


# ---------------------------------------------------------------------
# Attempt counter + restart semantics
# ---------------------------------------------------------------------

def test_re_entering_same_stage_bumps_attempt():
    lines = [
        "STAGE 1: calibration",
        "step=100 loss=5 text=5 loc=5 lr=1e-04",
        "supervisor: chain exited with code 1",
        "STAGE 1: calibration",
        "step=0 loss=10 text=10 loc=10 lr=1e-08",
    ]
    out, st = _run(lines)
    assert st.attempt == 2
    last_step = [r for r in out if r["event"] == "train_step"][-1]
    assert last_step["attempt"] == 2
    assert last_step["step"] == 0


def test_advancing_stage_does_not_bump_attempt():
    lines = [
        "STAGE 1: calibration",
        "step=10 loss=5 text=5 loc=5 lr=1e-04",
        "STAGE 2: multimodal",
        "step=0 loss=8 text=8 loc=8 lr=1e-08",
    ]
    _, st = _run(lines)
    assert st.attempt == 1
    assert st.stage == 2


def test_restart_alert_threshold():
    args = _default_args(restart_alert_threshold=3)
    lines = ["supervisor: chain exited with code 1"] * 4
    out, _ = _run(lines, args)
    restarts = [r for r in out if r["event"] == "supervisor_restart"]
    assert [r["restart_streak"] for r in restarts] == [1, 2, 3, 4]
    assert [r["alert"] for r in restarts] == [False, False, True, True]


# ---------------------------------------------------------------------
# Anomalies + gating
# ---------------------------------------------------------------------

def test_nan_loss_emits_anomaly():
    lines = ["step=42 loss=nan text=nan loc=nan lr=1e-04"]
    out, _ = _run(lines)
    kinds = [r["event"] for r in out]
    assert "train_step" in kinds and "nan_loss" in kinds


def test_nan_loss_serialises_as_null():
    """JSONL must be valid strict JSON -- NaN -> null."""
    sink = tailer_mod.JsonlSink(Path("/dev/null"), reset=True)
    sink._fp = __import__("io").StringIO()  # type: ignore[attr-defined]
    sink.write({"event": "train_step", "loss": float("nan")})
    line = sink._fp.getvalue().splitlines()[0]
    parsed = json.loads(line)  # strict JSON; would fail if NaN written literal
    assert parsed["loss"] is None


def test_oom_event():
    lines = ["torch.cuda.OutOfMemoryError: CUDA out of memory."]
    out, _ = _run(lines)
    assert out[0]["event"] == "oom"


def test_traceback_event():
    lines = [
        'Traceback (most recent call last):',
        '  File "x.py", line 1, in <module>',
        'RuntimeError: boom',
    ]
    out, _ = _run(lines)
    assert out[0]["event"] == "traceback"


def test_empty_streak_gated_stage1_no_alert():
    """empty=N/N at stage 1 must not trigger an alert (frozen-decoder noise)."""
    lines = [
        "STAGE 1: calibration",
        *[
            f"validation step={500 + i*500} val_loss=8.5 n=20 (8.0s) "
            f"cer=1.0 wer=1.0 word-f1=0.0 decoded=5 empty=5"
            for i in range(5)
        ],
    ]
    out, _ = _run(lines)
    assert not any(r["event"] == "empty_decode_streak" for r in out)


def test_empty_streak_alert_fires_in_stage2_after_threshold_step():
    args = _default_args(gate_empty_after_stage=2, gate_empty_after_step=5000,
                         empty_streak=3)
    lines = [
        "STAGE 2: multimodal",
        # First two empty-vals at step >= 5000 -- streak builds, no alert yet.
        "validation step=5000 val_loss=5.5 n=20 (8.0s) cer=1.0 wer=1.0 word-f1=0.0 decoded=5 empty=5",
        "validation step=6000 val_loss=5.5 n=20 (8.0s) cer=1.0 wer=1.0 word-f1=0.0 decoded=5 empty=5",
        # Third one trips the alert.
        "validation step=7000 val_loss=5.5 n=20 (8.0s) cer=1.0 wer=1.0 word-f1=0.0 decoded=5 empty=5",
    ]
    out, _ = _run(lines, args)
    alerts = [r for r in out if r["event"] == "empty_decode_streak"]
    assert len(alerts) == 1
    assert alerts[0]["streak"] == 3


# ---------------------------------------------------------------------
# TensorBoard path (guarded for CPU-only CI)
# ---------------------------------------------------------------------

def test_tb_writer_emits_scalars(tmp_path):
    pytest.importorskip("tensorboard")
    pytest.importorskip("torch.utils.tensorboard")
    tb = tailer_mod.TbSink(tmp_path / "tb")
    tb.add(stage=1, attempt=1, tag="train/loss", value=5.0, step=100)
    tb.add(stage=1, attempt=1, tag="train/loss", value=4.5, step=200)
    tb.close()
    # Stage1 (attempt 1) writes to logs/tb/stage1/
    assert (tmp_path / "tb" / "stage1").exists()


def test_tb_attempt_split(tmp_path):
    pytest.importorskip("tensorboard")
    pytest.importorskip("torch.utils.tensorboard")
    tb = tailer_mod.TbSink(tmp_path / "tb")
    tb.add(stage=1, attempt=2, tag="train/loss", value=4.0, step=10)
    tb.close()
    # Attempt >= 2 must write to a separate dir to avoid overlay zigzag.
    assert (tmp_path / "tb" / "stage1_attempt2").exists()
    assert not (tmp_path / "tb" / "stage1").exists()


def test_tb_skips_nan(tmp_path):
    pytest.importorskip("tensorboard")
    pytest.importorskip("torch.utils.tensorboard")
    tb = tailer_mod.TbSink(tmp_path / "tb")
    # Should not raise, should not create a writer for the NaN value alone.
    tb.add(stage=1, attempt=1, tag="train/loss", value=float("nan"), step=1)
    tb.close()


# ---------------------------------------------------------------------
# Append-mode JSONL
# ---------------------------------------------------------------------

def test_jsonl_append_default_preserves_existing(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text('{"event":"prior"}\n', encoding="utf-8")
    sink = tailer_mod.JsonlSink(p, reset=False)
    sink.write({"event": "new"})
    sink.close()
    lines = p.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line)["event"] for line in lines]
    assert events == ["prior", "new"]


def test_jsonl_reset_truncates(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text('{"event":"prior"}\n', encoding="utf-8")
    sink = tailer_mod.JsonlSink(p, reset=True)
    sink.write({"event": "fresh"})
    sink.close()
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "fresh"
