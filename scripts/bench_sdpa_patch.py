"""Step-0 benchmark: decide whether the C3 SDPA patch is worth keeping.

Runs the eager softmax-bmm and SDPA at the paper attention shape
(B=1, H=16, T=2048, D=64) and at a smaller shape, and additionally
times one full forward through ``MBartForCausalLM`` eager-vs-patched
end-to-end. Prints wall-clock + peak-memory delta.

Run on the L40s (or any CUDA box) before investing in the rest of
the C3 plan::

    python scripts/bench_sdpa_patch.py

Decision rule (per the plan):
  * ``speedup >= 1.10`` OR ``peak_mem <= -0.10`` -> keep the patch and
    proceed with the rest of the work.
  * otherwise -> ``git rm src/vista_ocr/models/sdpa_patch.py`` and the
    ``--sdpa`` flag from stage scripts. No patch beats a polished
    patch nobody uses.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from vista_ocr.logging_config import setup_logging
from vista_ocr.models.sdpa_patch import apply, benchmark, revert


def _bench_end_to_end(
    *,
    batch: int,
    seq: int,
    iters: int,
    dtype: torch.dtype,
    device: str,
    train: bool,
) -> dict[str, float]:
    """Time a full ``MBartForCausalLM`` forward (optionally + backward)
    eager vs patched. ``train=True`` runs forward + backward + zero
    grads which is what the actual training loop pays."""
    from transformers import MBartConfig
    from transformers.models.mbart.modeling_mbart import MBartForCausalLM

    config = MBartConfig(
        vocab_size=64,
        d_model=1024,
        decoder_layers=12,
        decoder_attention_heads=16,
        decoder_ffn_dim=4096,
        max_position_embeddings=seq + 16,
        is_decoder=True,
        add_cross_attention=False,
    )
    dev = torch.device(device)
    torch.manual_seed(0)
    model = MBartForCausalLM(config).to(dev).to(dtype)
    model.train(train)
    ids = torch.randint(0, config.vocab_size, (batch, seq), device=dev)
    labels = ids.clone()

    def _run() -> None:
        if train:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            out = model(input_ids=ids, labels=labels)
            out.loss.backward()
        else:
            with torch.no_grad():
                model(input_ids=ids).logits

    def _time() -> tuple[float, int]:
        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            _run()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _run()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / iters
        peak = (
            int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0
        )
        return elapsed, peak

    revert()
    eager_t, eager_mem = _time()
    apply()
    try:
        sdpa_t, sdpa_mem = _time()
    finally:
        revert()

    return {
        "eager_s": eager_t,
        "sdpa_s": sdpa_t,
        "speedup": eager_t / max(sdpa_t, 1e-9),
        "eager_peak_bytes": float(eager_mem),
        "sdpa_peak_bytes": float(sdpa_mem),
        "peak_mem_delta": (
            (sdpa_mem - eager_mem) / eager_mem if eager_mem else 0.0
        ),
    }


def _print_section(title: str, r: dict[str, float]) -> None:
    print(f"\n=== {title} ===")
    for k, v in r.items():
        if "bytes" in k:
            print(f"  {k:>20s}: {v / 1e6:.1f} MB")
        elif "delta" in k or k == "speedup":
            print(f"  {k:>20s}: {v:+.2%}" if "delta" in k else f"  {k:>20s}: {v:.2f}x")
        else:
            print(f"  {k:>20s}: {v * 1000:.2f} ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None,
                    help="cuda or cpu (default: cuda if available).")
    ap.add_argument("--end-to-end-seq", type=int, default=2048,
                    help="Sequence length for the end-to-end MBart bench.")
    ap.add_argument("--end-to-end-iters", type=int, default=5)
    ap.add_argument("--kernel-iters", type=int, default=20)
    args = ap.parse_args()

    setup_logging(level="WARNING")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype}")

    # 1) Kernel-only at paper shape.
    kernel_paper = benchmark(
        batch=1, heads=16, seq=2048, head_dim=64,
        iters=args.kernel_iters, dtype=dtype, device=device,
    )
    _print_section("kernel-only @ (1, 16, 2048, 64)", kernel_paper)

    # 2) Kernel-only at smaller shape (sanity check the speedup curve).
    kernel_small = benchmark(
        batch=1, heads=16, seq=512, head_dim=64,
        iters=args.kernel_iters, dtype=dtype, device=device,
    )
    _print_section("kernel-only @ (1, 16, 512, 64)", kernel_small)

    # 3) End-to-end MBartForCausalLM forward only (eval).
    e2e_eval = _bench_end_to_end(
        batch=1, seq=args.end_to_end_seq, iters=args.end_to_end_iters,
        dtype=dtype, device=device, train=False,
    )
    _print_section(
        f"end-to-end MBart fwd-only (eval) @ T={args.end_to_end_seq}",
        e2e_eval,
    )

    # 4) End-to-end MBartForCausalLM forward + backward (training).
    # Backward kernel is a different SDPA path; this is the number that
    # matters for training-time speedup.
    e2e_train = _bench_end_to_end(
        batch=1, seq=args.end_to_end_seq, iters=args.end_to_end_iters,
        dtype=dtype, device=device, train=True,
    )
    _print_section(
        f"end-to-end MBart fwd+bwd (train) @ T={args.end_to_end_seq}",
        e2e_train,
    )

    # Decision summary -- train numbers drive the keep/drop call.
    speedup = e2e_train["speedup"]
    mem = e2e_train["peak_mem_delta"]
    print("\n--- DECISION (based on fwd+bwd, the training-loop path) ---")
    print(f"speedup:   {speedup:.2f}x")
    print(f"peak-mem:  {mem:+.2%}")
    print(f"(eval-only speedup for reference: {e2e_eval['speedup']:.2f}x)")
    if speedup >= 1.10 or mem <= -0.10:
        print("KEEP the SDPA patch and proceed with the C3 plan.")
        sys.exit(0)
    print("CONSIDER DROPPING the SDPA patch -- gains are below the "
          "10% threshold on both axes.")
    sys.exit(1)


if __name__ == "__main__":
    main()
