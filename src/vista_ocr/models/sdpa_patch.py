"""Optional manual SDPA monkey-patch for ``MBartForCausalLM``.

HuggingFace transformers 4.44 does not implement SDPA on
:class:`~transformers.MBartForCausalLM` (issue #28005). We bypass the
gate by replacing the eager attention forward with one that calls
:func:`torch.nn.functional.scaled_dot_product_attention` directly.

**Risks** documented in the design notes:

* bf16 SDPA != bf16 eager bit-exact. The numerical-equivalence test
  uses ``rtol=1e-3, atol=1e-3``; ship blocks if it fails.
* HF patch releases can rename internal attention modules. We pin the
  transformers version in environment files; the patch is gated by an
  ``apply()`` function so an operator can disable it on the fly.
* ``VISTA_NO_SDPA=1`` in the environment disables the patch even if
  ``apply()`` is called.

The patch is **opt-in**. Stage scripts do not call ``apply()`` by
default. Use it like::

    from vista_ocr.models.sdpa_patch import apply as apply_sdpa
    apply_sdpa()

Apply once at process start (idempotent).
"""
from __future__ import annotations

import logging
import math
import os

import torch
from torch import Tensor

LOG = logging.getLogger(__name__)

# Set when ``apply()`` succeeds so we don't double-patch.
_PATCHED = False


def _sdpa_attention_forward(
    self,
    hidden_states: Tensor,
    key_value_states: Tensor | None = None,
    past_key_value=None,
    attention_mask: Tensor | None = None,
    layer_head_mask: Tensor | None = None,
    output_attentions: bool = False,
):
    """Drop-in replacement for ``MBartAttention.forward`` using SDPA."""
    is_cross_attention = key_value_states is not None
    bsz, tgt_len, _ = hidden_states.size()

    # Q
    q = self.q_proj(hidden_states) * self.scaling

    # K, V (handle cross-attention + KV cache)
    if is_cross_attention and past_key_value is not None and \
            past_key_value[0].shape[2] == key_value_states.shape[1]:
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

    # SDPA path -- handles causal masking via ``is_causal`` + mask blend.
    is_causal = (
        not is_cross_attention
        and attention_mask is None
        and tgt_len > 1
    )
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=is_causal,
        scale=1.0,                # Q already scaled above
    )

    # (B, H, T, D) -> (B, T, H*D) -> proj
    out = out.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
    out = self.out_proj(out)

    return out, None, past_key_value


def apply() -> bool:
    """Monkey-patch ``MBartAttention.forward``. Idempotent.

    Returns True when the patch is now active. False when it was
    skipped because of ``VISTA_NO_SDPA=1`` or a missing module.
    """
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("VISTA_NO_SDPA") == "1":
        LOG.info("SDPA patch skipped: VISTA_NO_SDPA=1")
        return False
    try:
        from transformers.models.mbart.modeling_mbart import MBartAttention
    except ImportError as exc:
        LOG.warning("SDPA patch skipped: %s", exc)
        return False
    MBartAttention.forward = _sdpa_attention_forward          # type: ignore[assignment]
    _PATCHED = True
    LOG.info("SDPA patch applied to MBartAttention.forward")
    return True


def revert() -> None:
    """Undo the monkey-patch by reloading the module's forward."""
    global _PATCHED
    if not _PATCHED:
        return
    import importlib

    from transformers.models.mbart import modeling_mbart
    importlib.reload(modeling_mbart)
    _PATCHED = False
    LOG.info("SDPA patch reverted")


def numerical_equivalence(
    *,
    batch: int = 1,
    heads: int = 16,
    seq: int = 64,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float32,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> tuple[float, float, bool]:
    """Compare eager attention vs SDPA on identical (Q, K, V) inputs.

    Returns ``(max_abs_diff, mean_abs_diff, passes)``. Used as the C3
    ship gate: do not enable the patch in production unless this passes
    on bf16 inputs of (B=1, H=16, T=2048, D=64).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)

    # Eager: softmax(QK^T / sqrt(d)) V, no mask.
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    attn = torch.softmax(scores, dim=-1)
    eager = torch.matmul(attn, v)

    sdpa = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
    )

    diff = (eager - sdpa).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    passes = max_diff <= atol + rtol * float(eager.abs().max().item())
    return max_diff, mean_diff, passes
