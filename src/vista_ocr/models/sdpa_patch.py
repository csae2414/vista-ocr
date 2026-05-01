"""Optional SDPA monkey-patch for ``MBartAttention`` (C3).

HuggingFace transformers 4.44.2 does not implement SDPA on
:class:`~transformers.MBartForCausalLM` (issue #28005). This module
replaces ``MBartAttention.forward`` with a drop-in equivalent that
calls :func:`torch.nn.functional.scaled_dot_product_attention`.

The patch is **opt-in**. Stage scripts do not call :func:`apply` by
default. Operators enable it via ``--sdpa`` on the stage scripts after
the ship gate (see below) passes.

Pinned transformers version
---------------------------

Tested against ``transformers==4.44.2``. Other versions: a structural
attribute check (:data:`_REQUIRED_ATTRS`) is the real safeguard --
:func:`apply` refuses to patch when any required attribute is missing.
A version mismatch alone produces a WARNING but does not block, so a
routine dep bump does not silently disable SDPA.

What is NOT supported
---------------------

* ``output_attentions=True`` -- raises :class:`NotImplementedError`.
  SDPA does not return attention weights. Stage scripts never set
  this; set it to ``False`` (the default) or call :func:`revert` first.

Ship gate
---------

Run::

    python -m vista_ocr.models.sdpa_patch --check

This executes the equivalence + benchmark checks and prints one
summary line on success::

    SDPA patch: PASS -- speedup 1.7x, peak-mem -22%, max_diff 4e-04

Stage scripts that pass ``--sdpa`` invoke this check first and abort
if it fails. Reproducibility: any CER/WER number reported in a paper-
comparison context produced with ``--sdpa`` enabled must cite the
``--check`` PASS line (transformers version + max_diff) so reviewers
can confirm the kernel swap was equivalence-tested for that run.

Step-0 measurement (go/no-go)
-----------------------------

Run :file:`scripts/bench_sdpa_patch.py` on the L40s once to confirm
the patch buys >=10% on speedup OR peak memory. Operator updates the
table below after each box change so the value of keeping this
module is transparent::

    DEVICE      DTYPE   SHAPE                  SPEEDUP   PEAK_MEM
    ----------- ------- ---------------------- --------- ----------
    L40s        bf16    (1, 16, 2048, 64)      <fill>    <fill>
    L40s e2e    bf16    MBartForCausalLM 12L   <fill>    <fill>

If either column drops below the 10% threshold on the operator's
target box, prefer ``git rm`` over polishing -- "no patch" beats "a
patch that does nothing."

Disabling / overriding
----------------------

* ``VISTA_NO_SDPA=1`` -- :func:`apply` returns False without patching
  (the patch never installs).
* ``VISTA_SDPA_SKIP_CHECK=1`` -- :func:`enable_with_ship_gate` skips
  the ship-gate but still installs the patch. Use only when you have
  already verified the gate on this machine + transformers version.
  A WARNING is logged.
* :func:`revert` -- restores the captured original ``forward`` method.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

LOG = logging.getLogger(__name__)

# Single source of truth: every private attr the patched forward
# touches on ``MBartAttention``. ``apply()`` iterates this list and
# refuses to patch if any are missing.
_REQUIRED_ATTRS: tuple[str, ...] = (
    "_shape",
    "scaling",
    "is_decoder",
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "dropout",
    "embed_dim",
    "num_heads",
    "head_dim",
)

# Versions we have run the ship gate against. Outside this set we WARN
# but proceed -- the structural assert is the hard safeguard.
_SUPPORTED_TRANSFORMERS: tuple[str, ...] = ("4.44.2",)

# Set when ``apply()`` succeeds.
_PATCHED: bool = False
# Captured at apply() time so revert() can restore without
# importlib.reload (which doesn't reach modules holding references).
_ORIGINAL_FORWARD: Callable | None = None


# ---------------------------------------------------------------------
# Pure helpers (unit-tested directly)
# ---------------------------------------------------------------------

def _mask_is_purely_causal(mask: Tensor | None) -> bool:
    """True iff ``mask`` is the canonical additive causal mask.

    HF builds an additive 4-D mask of shape ``(B, 1, T, T)`` where the
    upper triangle (above the diagonal) is the dtype's finfo.min and
    the lower triangle + diagonal are 0. When training combines this
    with a padding mask, lower-triangular positions also become -inf,
    so the result is no longer purely causal -- this function returns
    False and the caller falls back to ``attn_mask=`` instead of
    ``is_causal=True``.
    """
    if mask is None or mask.ndim != 4:
        return False
    _b, _h, t, s = mask.shape
    if t != s or t == 0:
        return False
    neg_inf = torch.finfo(mask.dtype).min
    expected = torch.zeros(t, t, dtype=mask.dtype, device=mask.device)
    upper = torch.triu(
        torch.ones(t, t, dtype=torch.bool, device=mask.device), diagonal=1,
    )
    expected.masked_fill_(upper, neg_inf)
    # Broadcast-compare: the canonical mask is (T, T); HF emits it
    # broadcast to (B, 1, T, T). torch.equal won't broadcast, so we
    # check (mask == expected) elementwise after broadcasting.
    return bool(torch.equal(mask.expand(_b, _h, t, t), expected.expand(_b, _h, t, t)))


# ---------------------------------------------------------------------
# Patched forward
# ---------------------------------------------------------------------

def _sdpa_attention_forward(
    self,
    hidden_states: Tensor,
    key_value_states: Tensor | None = None,
    past_key_value: tuple[Tensor, Tensor] | None = None,
    attention_mask: Tensor | None = None,
    layer_head_mask: Tensor | None = None,
    output_attentions: bool = False,
):
    """Drop-in replacement for ``MBartAttention.forward`` using SDPA.

    Numerically equivalent to the eager forward within float tolerance
    (see :func:`equivalence_report`). When ``attention_mask`` is the
    canonical causal mask we drop it and pass ``is_causal=True`` so
    SDPA can use its fused causal kernel.
    """
    if output_attentions:
        raise NotImplementedError(
            "SDPA patch does not support output_attentions=True. "
            "Set output_attentions=False or call sdpa_patch.revert() first.",
        )
    if layer_head_mask is not None:
        raise NotImplementedError(
            "SDPA patch does not support layer_head_mask. "
            "Call sdpa_patch.revert() if you need head-masking.",
        )

    is_cross_attention = key_value_states is not None
    bsz, tgt_len, _ = hidden_states.size()

    q = self.q_proj(hidden_states) * self.scaling

    if (
        is_cross_attention
        and past_key_value is not None
        and past_key_value[0].shape[2] == key_value_states.shape[1]
    ):
        k = past_key_value[0]
        v = past_key_value[1]
    elif is_cross_attention:
        k = self._shape(self.k_proj(key_value_states), -1, bsz)
        v = self._shape(self.v_proj(key_value_states), -1, bsz)
    elif past_key_value is not None:
        k = self._shape(self.k_proj(hidden_states), -1, bsz)
        v = self._shape(self.v_proj(hidden_states), -1, bsz)
        k = torch.cat([past_key_value[0], k], dim=2)
        v = torch.cat([past_key_value[1], v], dim=2)
    else:
        k = self._shape(self.k_proj(hidden_states), -1, bsz)
        v = self._shape(self.v_proj(hidden_states), -1, bsz)

    if self.is_decoder:
        past_key_value = (k, v)

    q = self._shape(q, tgt_len, bsz)

    # Convert "additive purely-causal mask" to is_causal=True so SDPA
    # uses its fused causal kernel during training. Anything else
    # (padding combined with causal, encoder bidir mask, cross-attn
    # mask) is passed through as attn_mask.
    if _mask_is_purely_causal(attention_mask):
        sdpa_mask: Tensor | None = None
        is_causal = True
    else:
        sdpa_mask = attention_mask
        is_causal = False

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=sdpa_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=is_causal,
        scale=1.0,  # q already pre-scaled by self.scaling above
    )

    out = out.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
    out = self.out_proj(out)

    return out, None, past_key_value


# ---------------------------------------------------------------------
# Apply / revert
# ---------------------------------------------------------------------

def _check_required_attrs(cls: type) -> list[str]:
    """Return the list of missing required attrs on a sample instance.

    Builds a tiny instance because some attrs are set in ``__init__``,
    not on the class.
    """
    try:
        instance = cls(embed_dim=8, num_heads=2, dropout=0.0, is_decoder=True)
    except Exception:  # pragma: no cover - constructor signature change
        # Fall back to class-level check if construction fails.
        return [a for a in _REQUIRED_ATTRS if not hasattr(cls, a)]
    return [a for a in _REQUIRED_ATTRS if not hasattr(instance, a)]


def apply() -> bool:
    """Monkey-patch ``MBartAttention.forward``. Idempotent.

    Returns True when the patch is now active. False when skipped
    because of ``VISTA_NO_SDPA=1``, missing import, or a structural
    attribute check failure.
    """
    global _PATCHED, _ORIGINAL_FORWARD
    if _PATCHED:
        return True
    if os.environ.get("VISTA_NO_SDPA") == "1":
        LOG.info("SDPA patch skipped: VISTA_NO_SDPA=1")
        return False
    try:
        import transformers
        from transformers.models.mbart.modeling_mbart import MBartAttention
    except ImportError as exc:
        LOG.warning("SDPA patch skipped: %s", exc)
        return False

    missing = _check_required_attrs(MBartAttention)
    if missing:
        LOG.error(
            "SDPA patch refused: MBartAttention is missing required "
            "attributes %s -- transformers internals may have changed.",
            missing,
        )
        return False

    if transformers.__version__ not in _SUPPORTED_TRANSFORMERS:
        LOG.warning(
            "SDPA patch: transformers==%s is outside the tested set %s; "
            "structural attrs are present so the patch will be applied. "
            "Run `python -m vista_ocr.models.sdpa_patch --check` to verify.",
            transformers.__version__, _SUPPORTED_TRANSFORMERS,
        )

    _ORIGINAL_FORWARD = MBartAttention.forward
    MBartAttention.forward = _sdpa_attention_forward  # type: ignore[assignment]
    _PATCHED = True
    LOG.info("SDPA patch applied to MBartAttention.forward")
    return True


def revert() -> None:
    """Restore the captured original ``forward``. Idempotent."""
    global _PATCHED, _ORIGINAL_FORWARD
    if not _PATCHED:
        return
    from transformers.models.mbart.modeling_mbart import MBartAttention
    MBartAttention.forward = _ORIGINAL_FORWARD  # type: ignore[assignment]
    _ORIGINAL_FORWARD = None
    _PATCHED = False
    LOG.info("SDPA patch reverted")


def call_eager(attn: nn.Module, *args, **kwargs):
    """Call the saved original ``forward`` on a (possibly patched) instance.

    Tests use this to compare patched vs eager outputs side-by-side
    without flipping the patch off and on between calls.
    """
    if _ORIGINAL_FORWARD is None:
        raise RuntimeError(
            "call_eager requires apply() to have been called first "
            "(it captures the original forward at that time).",
        )
    return _ORIGINAL_FORWARD(attn, *args, **kwargs)


# ---------------------------------------------------------------------
# Reusable equivalence + benchmark helpers (used by tests AND CLI)
# ---------------------------------------------------------------------

def _max_abs(a: Tensor, b: Tensor) -> float:
    return float((a - b).abs().max().item())


def equivalence_report(
    attn: nn.Module,
    fwd_kwargs: dict[str, Any],
    *,
    check_grads: bool = False,
    grad_params: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj"),
) -> dict[str, Any]:
    """Run patched + eager forward (and optional backward) on identical inputs.

    Requires :func:`apply` to have been called. Resets RNG between
    runs so dropout produces matching draws when ``training=True``.
    Returns a dict with forward diff, optional grad diffs, and pass
    flags.
    """
    if not _PATCHED:
        raise RuntimeError("equivalence_report requires apply() first")

    def _zero_grads() -> None:
        for p in attn.parameters():
            if p.grad is not None:
                p.grad = None

    # ---- patched ----
    _zero_grads()
    torch.manual_seed(0)
    out_p, _, kv_p = attn(**fwd_kwargs)
    grads_p: dict[str, Tensor] = {}
    if check_grads:
        out_p.sum().backward()
        for name in grad_params:
            mod = getattr(attn, name)
            grads_p[name] = mod.weight.grad.detach().clone()

    # ---- eager ----
    _zero_grads()
    torch.manual_seed(0)
    out_e, _, kv_e = call_eager(attn, **fwd_kwargs)
    grads_e: dict[str, Tensor] = {}
    if check_grads:
        out_e.sum().backward()
        for name in grad_params:
            mod = getattr(attn, name)
            grads_e[name] = mod.weight.grad.detach().clone()

    fwd_diff = _max_abs(out_p, out_e)
    kv_diff: float | None = None
    if kv_p is not None and kv_e is not None:
        kv_diff = max(_max_abs(kv_p[0], kv_e[0]), _max_abs(kv_p[1], kv_e[1]))

    grad_diffs: dict[str, float] = {}
    if check_grads:
        for name in grad_params:
            grad_diffs[name] = _max_abs(grads_p[name], grads_e[name])

    return {
        "fwd_diff": fwd_diff,
        "kv_diff": kv_diff,
        "grad_diffs": grad_diffs if check_grads else None,
    }


def benchmark(
    *,
    batch: int = 1,
    heads: int = 16,
    seq: int = 2048,
    head_dim: int = 64,
    iters: int = 20,
    dtype: torch.dtype = torch.bfloat16,
    device: str | None = None,
) -> dict[str, float]:
    """Wall-clock + peak-memory eager vs patched on (B, H, T, D).

    Times only the attention kernel call (not surrounding projections)
    so the speedup number reflects what SDPA actually buys us.

    On CPU the bf16 path is supported but slow; the meaningful number
    is from a CUDA box.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq, head_dim, device=dev, dtype=dtype)
    k = torch.randn(batch, heads, seq, head_dim, device=dev, dtype=dtype)
    v = torch.randn(batch, heads, seq, head_dim, device=dev, dtype=dtype)

    def _eager() -> Tensor:
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def _sdpa() -> Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
        )

    def _time(fn: Callable[[], Tensor]) -> tuple[float, int]:
        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        # warmup
        for _ in range(3):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / iters
        peak = (
            int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0
        )
        return elapsed, peak

    eager_t, eager_mem = _time(_eager)
    sdpa_t, sdpa_mem = _time(_sdpa)
    return {
        "eager_s_per_iter": eager_t,
        "sdpa_s_per_iter": sdpa_t,
        "speedup": eager_t / max(sdpa_t, 1e-9),
        "eager_peak_bytes": float(eager_mem),
        "sdpa_peak_bytes": float(sdpa_mem),
        "peak_mem_delta": (
            (sdpa_mem - eager_mem) / eager_mem if eager_mem else 0.0
        ),
    }


# ---------------------------------------------------------------------
# CLI ship gate
# ---------------------------------------------------------------------

def run_check_cli(
    *,
    atol_fwd: float = 5e-3,
    atol_grad: float = 5e-3,
    atol_bf16: float = 5e-3,
) -> int:
    """Run every equivalence test we can run on the current device.

    Prints one summary line on success carrying BOTH the fp32
    forward+grad diff AND the bf16 paper-shape diff -- operators
    paste this verbatim into PR descriptions and paper-comparison
    notes. Exits with status 1 on any failure.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        import transformers
        from transformers.models.mbart.modeling_mbart import MBartAttention
    except ImportError as exc:
        print(f"SDPA patch: FAIL -- transformers import failed: {exc}",
              file=sys.stderr)
        return 1

    if not apply():
        print("SDPA patch: FAIL -- apply() returned False", file=sys.stderr)
        return 1

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- fp32 forward + grad on a small layer ----
        bsz, tgt_len, embed_dim, num_heads = 2, 16, 64, 4
        attn = MBartAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=0.0, is_decoder=True,
        ).to(device).eval()
        torch.manual_seed(0)
        hidden = torch.randn(bsz, tgt_len, embed_dim, device=device)
        rep_self = equivalence_report(
            attn, {"hidden_states": hidden}, check_grads=True,
        )
        if rep_self["fwd_diff"] > atol_fwd:
            print(f"SDPA patch: FAIL -- fp32 self-attn fwd diff "
                  f"{rep_self['fwd_diff']:.2e} > {atol_fwd:.0e}",
                  file=sys.stderr)
            return 1
        for name, d in rep_self["grad_diffs"].items():
            if d > atol_grad:
                print(f"SDPA patch: FAIL -- fp32 self-attn grad[{name}] "
                      f"{d:.2e} > {atol_grad:.0e}", file=sys.stderr)
                return 1

        # ---- bf16 paper-shape forward (the actual ship gate) ----
        bf16_dim = 1024
        bf16_attn = MBartAttention(
            embed_dim=bf16_dim, num_heads=16, dropout=0.0, is_decoder=True,
        ).to(device).to(torch.bfloat16).eval()
        torch.manual_seed(0)
        bf16_hidden = torch.randn(
            1, 2048, bf16_dim, device=device, dtype=torch.bfloat16,
        )
        rep_bf16 = equivalence_report(
            bf16_attn, {"hidden_states": bf16_hidden}, check_grads=False,
        )
        if rep_bf16["fwd_diff"] > atol_bf16:
            print(f"SDPA patch: FAIL -- bf16 paper-shape fwd diff "
                  f"{rep_bf16['fwd_diff']:.2e} > {atol_bf16:.0e}",
                  file=sys.stderr)
            return 1

        # ---- benchmark ----
        bench = benchmark(
            batch=1, heads=16, seq=512, head_dim=64,
            iters=10,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device=device.type,
        )
    finally:
        revert()

    print(
        f"SDPA patch: PASS -- speedup {bench['speedup']:.2f}x, "
        f"peak-mem {bench['peak_mem_delta']:+.0%}, "
        f"fp32_diff {rep_self['fwd_diff']:.2e}, "
        f"bf16_diff {rep_bf16['fwd_diff']:.2e} "
        f"(transformers {transformers.__version__})",
    )
    return 0


def enable_with_ship_gate(*, manifest_dir: "os.PathLike[str] | None" = None) -> None:
    """Stage-script entrypoint: run the ship-gate, then activate SDPA.

    Aborts via :func:`sys.exit` if the ship-gate fails so a failed
    kernel never silently feeds a real training run.

    ``VISTA_SDPA_SKIP_CHECK=1`` skips the ship-gate (operator already
    verified on this machine + transformers version). A WARNING is
    logged so accidental skips show up in stage logs.

    ``manifest_dir`` (typically ``args.out``): when provided AND the
    ship-gate ran, the PASS line is written to
    ``<manifest_dir>/sdpa_manifest.txt``. Paper-comparison runs cite
    this file so reviewers can confirm the kernel swap was equivalence-
    tested for that exact run.
    """
    skip = os.environ.get("VISTA_SDPA_SKIP_CHECK") == "1"
    pass_line: str | None = None

    if skip:
        LOG.warning(
            "VISTA_SDPA_SKIP_CHECK=1 -- skipping ship-gate. The operator "
            "is responsible for having verified `python -m "
            "vista_ocr.models.sdpa_patch --check` on this machine + "
            "transformers version.",
        )
    else:
        # Capture the PASS line so we can pin it into the run manifest.
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = run_check_cli()
        sys.stdout.write(buf.getvalue())
        if rc != 0:
            sys.exit("SDPA ship-gate failed; aborting training run.")
        pass_line = buf.getvalue().strip().splitlines()[-1] if buf.getvalue() else None

    if not apply():
        sys.exit("SDPA apply() returned False after a passing ship-gate "
                 "-- this should not happen; investigate before retrying.")

    if manifest_dir is not None and pass_line:
        from pathlib import Path
        manifest = Path(manifest_dir) / "sdpa_manifest.txt"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(pass_line + "\n", encoding="utf-8")
        LOG.info("SDPA manifest written to %s", manifest)

    LOG.info("SDPA patch active for this training run.")


def _main() -> None:
    ap = argparse.ArgumentParser(prog="vista_ocr.models.sdpa_patch")
    ap.add_argument(
        "--check", action="store_true",
        help="Run the ship-gate equivalence + benchmark and exit.",
    )
    args = ap.parse_args()
    if args.check:
        sys.exit(run_check_cli())
    ap.print_help()
    sys.exit(2)


if __name__ == "__main__":
    _main()
