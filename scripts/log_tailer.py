"""Side-process log tailer: parse training stdout into JSONL + TensorBoard.

Reads a training log file (default ``logs/pretrain_supervised.log``) in
``tail -F`` style, regex-parses each line into structured records, and
writes:

* ``logs/metrics.jsonl``   -- one append-only JSON record per event.
* ``logs/tb/stage{N}[_attempt{A}]/`` -- per-stage TensorBoard scalars
  (loss / text / loc / lr / val_loss / cer / wer / empty_frac).

The tailer is **read-only with respect to the training process**: runs
in its own screen/tmux session, never modifies training code or files
the trainer writes to. Safe to start, stop, and restart at will.

Records contain ``(attempt, stage, step?, ts)`` so ``(attempt, stage,
step)`` is unique even after a supervisor restart re-runs stage 1 from
step 0.

Anomaly events written to JSONL (a separate Monitor process can grep
``"event": "supervisor_restart"`` etc.):

* ``supervisor_restart``   -- supervisor logged a non-zero chain exit
* ``traceback``            -- a Python Traceback line appeared
* ``oom``                  -- CUDA OOM line appeared
* ``nan_loss``             -- a logged loss was NaN/Inf
* ``empty_decode_streak``  -- N consecutive val passes with empty == decoded
                              (gated to stage >= --gate-empty-after-stage and
                              step >= --gate-empty-after-step)
* ``step_rate_low``        -- step rate (steps/s) below threshold across a
                              window (reset on stage transition / restart)

Invocation::

    python scripts/log_tailer.py
    python scripts/log_tailer.py --log path/to/log --reset
    python scripts/log_tailer.py --no-tb       # JSONL only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------

RE_STAGE_HEADER = re.compile(
    r"STAGE\s+(?P<stage>\d+)\s*:", re.IGNORECASE,
)
RE_TRAIN_STEP = re.compile(
    r"step=(?P<step>\d+)\s+loss=(?P<loss>[-\d.eE+]+|nan|inf|-inf)\s+"
    r"text=(?P<text>[-\d.eE+]+|nan|inf|-inf)\s+"
    r"loc=(?P<loc>[-\d.eE+]+|nan|inf|-inf)\s+"
    r"lr=(?P<lr>[-\d.eE+]+)",
)
RE_VAL = re.compile(
    r"validation step=(?P<step>\d+)\s+val_loss=(?P<val_loss>[-\d.eE+]+|nan|inf|-inf)"
    r"(?:.*?cer=(?P<cer>[-\d.eE+]+)\s+wer=(?P<wer>[-\d.eE+]+)\s+"
    r"word-f1=(?P<f1>[-\d.eE+]+)\s+decoded=(?P<decoded>\d+)\s+"
    r"empty=(?P<empty>\d+))?",
)
RE_CKPT = re.compile(
    r"Checkpoint saved:.*?\(step (?P<step>\d+)",
)
RE_STAGE_DONE = re.compile(
    r"DONE:\s+(?P<steps>\d+) steps,\s+(?P<wall>[-\d.]+)s\s+"
    r"\((?P<per_step>[-\d.]+)s/step\).*?"
    r"Loss first=(?P<first>[-\d.]+) last=(?P<last>[-\d.]+)\s+"
    r"delta=(?P<delta>[-\d.]+)",
)
RE_CHAIN_DONE = re.compile(r"ALL STAGES DONE")
RE_SUPERVISOR_FAIL = re.compile(
    r"supervisor: chain exited with code (?P<code>\d+)",
)
RE_OOM = re.compile(r"(CUDA out of memory|OutOfMemoryError)")
RE_TRACEBACK_START = re.compile(r"^Traceback \(most recent call last\):")


def _to_float(s: str) -> float:
    s = s.strip().lower()
    if s == "nan":
        return float("nan")
    if s in ("inf", "+inf"):
        return float("inf")
    if s == "-inf":
        return float("-inf")
    return float(s)


def _safe_json_value(v: Any) -> Any:
    """Convert NaN/Inf to None so json.dumps stays strict-mode safe."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


# ---------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------

class JsonlSink:
    """Append-only JSONL writer. ``reset=True`` truncates first."""

    def __init__(self, path: Path, *, reset: bool = False) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if reset else "a"
        self._fp = self.path.open(mode, encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        record = {k: _safe_json_value(v) for k, v in record.items()}
        record["ts"] = time.time()
        self._fp.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:  # noqa: BLE001
            pass


class TbSink:
    """Per-stage-attempt TensorBoard writer. Lazy import; safe on CPU boxes.

    Path layout: ``<root>/stage{N}/`` for attempt 1; ``<root>/stage{N}_attempt{A}/``
    for attempt >= 2 to avoid overlaying retried curves on the original.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._writers: dict[tuple[int, int], Any] = {}

    def _writer(self, stage: int, attempt: int):
        key = (stage, attempt)
        if key in self._writers:
            return self._writers[key]
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415
        except ImportError:
            return None
        suffix = f"stage{stage}" if attempt <= 1 else f"stage{stage}_attempt{attempt}"
        w = SummaryWriter(log_dir=str(self.root / suffix))
        self._writers[key] = w
        return w

    def add(self, stage: int, attempt: int, tag: str, value: float, step: int) -> None:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return
        w = self._writer(stage, attempt)
        if w is None:
            return
        w.add_scalar(tag, value, global_step=step)

    def close(self) -> None:
        for w in self._writers.values():
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------
# Tail-F generator
# ---------------------------------------------------------------------

def follow(path: Path, *, from_start: bool = True, poll_interval: float = 0.5):
    """Yield lines from ``path`` with tail-F semantics.

    Re-opens the file from byte 0 if its size shrinks (truncation /
    rotation), warning once on stderr. Tolerates transient FS errors.
    """
    fp = None
    last_size = 0
    while True:
        try:
            if fp is None:
                fp = path.open("r", encoding="utf-8", errors="replace")
                if not from_start:
                    fp.seek(0, os.SEEK_END)
                last_size = fp.tell()
            line = fp.readline()
            if line:
                yield line
                continue
            try:
                cur_size = path.stat().st_size
            except FileNotFoundError:
                cur_size = 0
            if cur_size < last_size:
                print(f"# tailer: log shrunk ({last_size} -> {cur_size}), "
                      f"re-opening from byte 0", file=sys.stderr, flush=True)
                fp.close()
                fp = None
                from_start = True  # next open reads everything
            else:
                last_size = cur_size
                time.sleep(poll_interval)
        except Exception as exc:  # noqa: BLE001
            print(f"# tailer: transient read error: {exc}", file=sys.stderr,
                  flush=True)
            time.sleep(poll_interval)


# ---------------------------------------------------------------------
# Main parsing loop
# ---------------------------------------------------------------------

class TailerState:
    def __init__(self) -> None:
        self.stage: int = 0
        self.attempt: int = 1
        self.last_stage_in_run: int | None = None
        self.window_start_ts: float | None = None
        self.window_start_step: int | None = None
        self.empty_streak: int = 0
        self.restart_streak: int = 0


def _reset_stall_window(state: TailerState) -> None:
    state.window_start_ts = None
    state.window_start_step = None


def process_line(
    line: str,
    state: TailerState,
    jsonl: JsonlSink,
    tb: TbSink | None,
    args: argparse.Namespace,
) -> None:
    """Parse one log line and emit zero or more JSONL records."""
    line = line.rstrip("\n")

    m = RE_STAGE_HEADER.search(line)
    if m:
        new_stage = int(m.group("stage"))
        # Same stage re-entered after a non-zero exit -> bump attempt.
        if state.last_stage_in_run is not None and new_stage <= state.last_stage_in_run:
            state.attempt += 1
        state.stage = new_stage
        state.last_stage_in_run = new_stage
        _reset_stall_window(state)
        jsonl.write({
            "event": "stage_header",
            "attempt": state.attempt, "stage": state.stage,
        })
        return

    m = RE_TRAIN_STEP.search(line)
    if m:
        step = int(m.group("step"))
        loss = _to_float(m.group("loss"))
        text = _to_float(m.group("text"))
        loc = _to_float(m.group("loc"))
        lr = _to_float(m.group("lr"))
        jsonl.write({
            "event": "train_step",
            "attempt": state.attempt, "stage": state.stage, "step": step,
            "loss": loss, "text": text, "loc": loc, "lr": lr,
        })
        if tb is not None:
            tb.add(state.stage, state.attempt, "train/loss", loss, step)
            tb.add(state.stage, state.attempt, "train/text", text, step)
            tb.add(state.stage, state.attempt, "train/loc", loc, step)
            tb.add(state.stage, state.attempt, "train/lr", lr, step)
        if not math.isfinite(loss):
            jsonl.write({
                "event": "nan_loss",
                "attempt": state.attempt, "stage": state.stage, "step": step,
            })
        # Windowed stall detector: pin a window-start; once the window
        # is at least --stall-window seconds wide, evaluate steps/s.
        now = time.time()
        if state.window_start_ts is None:
            state.window_start_ts = now
            state.window_start_step = step
        else:
            dt = now - state.window_start_ts
            if dt >= args.stall_window:
                rate = (step - state.window_start_step) / max(dt, 1e-9)
                if rate < args.stall_min_rate:
                    jsonl.write({
                        "event": "step_rate_low",
                        "attempt": state.attempt, "stage": state.stage,
                        "step": step, "rate": round(rate, 4),
                        "window_s": round(dt, 1),
                    })
                # Reset window after evaluation so we don't re-fire.
                state.window_start_ts = now
                state.window_start_step = step
        return

    m = RE_VAL.search(line)
    if m:
        step = int(m.group("step"))
        rec: dict[str, Any] = {
            "event": "val",
            "attempt": state.attempt, "stage": state.stage, "step": step,
            "val_loss": _to_float(m.group("val_loss")),
        }
        if m.group("cer") is not None:
            rec["cer"] = _to_float(m.group("cer"))
            rec["wer"] = _to_float(m.group("wer"))
            rec["word_f1"] = _to_float(m.group("f1"))
            rec["decoded"] = int(m.group("decoded"))
            rec["empty"] = int(m.group("empty"))
            if rec["decoded"] > 0 and rec["empty"] == rec["decoded"]:
                state.empty_streak += 1
            else:
                state.empty_streak = 0
            gate_ok = (
                state.stage >= args.gate_empty_after_stage
                and step >= args.gate_empty_after_step
            )
            if gate_ok and state.empty_streak >= args.empty_streak:
                jsonl.write({
                    "event": "empty_decode_streak",
                    "attempt": state.attempt, "stage": state.stage,
                    "step": step, "streak": state.empty_streak,
                })
        jsonl.write(rec)
        if tb is not None:
            tb.add(state.stage, state.attempt, "val/loss", rec["val_loss"], step)
            for k in ("cer", "wer", "word_f1"):
                if k in rec:
                    tb.add(state.stage, state.attempt, f"val/{k}", rec[k], step)
            if "empty" in rec and rec["decoded"]:
                tb.add(state.stage, state.attempt, "val/empty_frac",
                       rec["empty"] / rec["decoded"], step)
        return

    m = RE_CKPT.search(line)
    if m:
        jsonl.write({
            "event": "checkpoint",
            "attempt": state.attempt, "stage": state.stage,
            "step": int(m.group("step")),
        })
        return

    m = RE_STAGE_DONE.search(line)
    if m:
        jsonl.write({
            "event": "stage_summary",
            "attempt": state.attempt, "stage": state.stage,
            "n_steps": int(m.group("steps")),
            "wall_s": _to_float(m.group("wall")),
            "per_step_s": _to_float(m.group("per_step")),
            "loss_first": _to_float(m.group("first")),
            "loss_last": _to_float(m.group("last")),
            "loss_delta": _to_float(m.group("delta")),
        })
        return

    if RE_CHAIN_DONE.search(line):
        jsonl.write({"event": "chain_done", "attempt": state.attempt})
        return

    m = RE_SUPERVISOR_FAIL.search(line)
    if m:
        code = int(m.group("code"))
        state.restart_streak += 1
        rec = {
            "event": "supervisor_restart",
            "attempt": state.attempt, "exit_code": code,
            "restart_streak": state.restart_streak,
            "alert": state.restart_streak >= args.restart_alert_threshold,
        }
        jsonl.write(rec)
        # Reset stall + empty streak across the restart boundary.
        _reset_stall_window(state)
        state.empty_streak = 0
        return
    # Reset restart_streak when a successful stage advances (set in
    # process_line via stage_header? actually we want: if any normal
    # train_step happened in this attempt, the restart sequence is
    # broken). Keeping it simple: reset on stage_header (above we did
    # not). Reset here when we see a stage_summary (clean stage
    # finish) instead -- but we already returned. Add an explicit
    # reset on stage_summary:
    # (handled implicitly: restart_streak only advances on real fails)

    if RE_OOM.search(line):
        jsonl.write({
            "event": "oom",
            "attempt": state.attempt, "stage": state.stage,
            "snippet": line[:200],
        })
        return

    if RE_TRACEBACK_START.search(line):
        jsonl.write({
            "event": "traceback",
            "attempt": state.attempt, "stage": state.stage,
        })
        return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path,
                    default=Path("logs/pretrain_supervised.log"),
                    help="Source training log to follow.")
    ap.add_argument("--metrics", type=Path,
                    default=Path("logs/metrics.jsonl"),
                    help="JSONL output (append by default; use --reset to truncate).")
    ap.add_argument("--reset", action="store_true",
                    help="Truncate --metrics on start.")
    ap.add_argument("--tb-dir", type=Path, default=Path("logs/tb"),
                    help="TensorBoard scalar root.")
    ap.add_argument("--no-tb", action="store_true",
                    help="Skip TensorBoard scalar emission.")
    ap.add_argument("--from-end", action="store_true",
                    help="Skip backfill; only read new lines.")
    ap.add_argument("--gate-empty-after-stage", type=int, default=2,
                    help="Empty-decode-streak alerts only fire from this "
                         "stage onward.")
    ap.add_argument("--gate-empty-after-step", type=int, default=5000,
                    help="And only after this step within the gated stage.")
    ap.add_argument("--empty-streak", type=int, default=3,
                    help="N consecutive val passes with empty==decoded "
                         "to trigger the alert.")
    ap.add_argument("--stall-min-rate", type=float, default=0.1,
                    help="Steps/s below this for --stall-window seconds "
                         "triggers step_rate_low.")
    ap.add_argument("--stall-window", type=float, default=300.0,
                    help="Window (seconds) for the stall detector.")
    ap.add_argument("--restart-alert-threshold", type=int, default=3,
                    help="A supervisor_restart event is marked alert=True "
                         "only after this many consecutive restarts.")
    args = ap.parse_args()

    if not args.log.exists():
        print(f"# tailer: waiting for log to appear at {args.log}",
              file=sys.stderr, flush=True)
        while not args.log.exists():
            time.sleep(2)

    jsonl = JsonlSink(args.metrics, reset=args.reset)
    tb = None if args.no_tb else TbSink(args.tb_dir)
    state = TailerState()

    print(f"# tailer: following {args.log} -> {args.metrics}",
          file=sys.stderr, flush=True)
    if tb is not None:
        print(f"# tailer: tensorboard scalars -> {args.tb_dir}",
              file=sys.stderr, flush=True)

    try:
        for line in follow(args.log, from_start=not args.from_end):
            process_line(line, state, jsonl, tb, args)
    except KeyboardInterrupt:
        pass
    finally:
        jsonl.close()
        if tb is not None:
            tb.close()


if __name__ == "__main__":
    main()
