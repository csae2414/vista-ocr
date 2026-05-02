# Benchmarks

Reproduction of the headline numbers from the original paper. Each row
is filled in after running `scripts/finetune_chain.sh` (which produces
the metrics via `scripts/finetune_eval.py` and a per-dataset log under
`logs/`).

| Dataset | Metric | Paper | Ours | Δ |
|---|---|---|---|---|
| SROIE 2019 | word-F1 | **93.95** | _TBD_ | _TBD_ |
| IAM        | WER     | **10.14** | _TBD_ | _TBD_ |
| MAURDOR-EN | Area-F1 | **87.02** | _TBD_ | _TBD_ |

Headline rows depend on licence-restricted datasets (SROIE, IAM,
MAURDOR) that are not bundled with the repository. PRs welcome.

## Pretraining run, 2026-05-02 (RTX 3090, 24 GB)

End-of-stage **train** loss is the trailing 50-step mean printed by
`scripts/stage{1,2,3}_run.py`. End-of-stage **val** loss is the last
validation pass before the stage's `DONE:` line, evaluated on a single
held-out PDFA shard (20 batches, batch-size 1).

| Stage | Steps | Wall (s) | Train loss (last 50) | Val loss (last pass) | Δ vs init |
|---|---|---|---|---|---|
| Stage-1 calibration | 20 000 | 2 890 | 8.836 | 8.590 | −1.20 |
| Stage-2 multimodal  | 80 000 | 11 697 | 4.735 | 4.315 | −4.64 |
| Stage-3 multitask   | 70 000 | 10 858 | 3.721 | 4.467 | −0.78 |
| **Total**           | **170 000** | **25 445 (~7 h 04 min)** | — | — | — |

Stage-3 val loss is higher than stage-2's because the val pass measures
only the OCR+layout task while stage-3 trains on a four-task mix.
`ckpt_best.pt` for stage-3 was saved at step 58 000 with val_loss
4.456.

## PDFA hold-out: greedy generation eval

Long-form eval on the same held-out PDFA shard, 100 batches. Run with:

```
python scripts/eval_pdfa_holdout.py \
  --ckpt checkpoints/stage3/ckpt_best.pt \
  --val-shard data/raw/pdfa/pdfa-eng-train-0119.tar \
  --spm data/processed/vocab/sp_en_16k.model \
  --max-batches 100 \
  --repetition-penalty 1.3 --no-repeat-ngram-size 6
```

| Metric | Value |
|---|---|
| Decoded batches | 100 |
| Empty hypotheses | 7 / 100 (7 %) |
| CER | **0.8414** |
| WER | 1.0218 |
| word-F1 | 0.1562 |
| Wall | 47 s |

**Honest read.** Numbers are weak relative to the paper's finetune
targets, but the model is no longer broken: outputs are image-
conditioned, structured (~30 `<x><y>` lines per page), and contain
real document text. The WER > 1.0 indicates insertion-dominated errors
— the model emits more text than the reference; this is consistent
with under-training (170 K steps, PDFA-only, no licence-restricted
finetune) and with the language-model prior still partially dominating
the encoder signal. Stronger conditioning would likely come from
(a) more pretraining steps, (b) finetune on the target dataset, or
(c) a stronger encoder→decoder bridge (the paper's exact decoder
init may be relevant here).

## Inference bug fixed during this run

The first hold-out eval reported 100 / 100 empty hypotheses despite
healthy training loss. Diagnostic dump (`--debug-dump 2`) found that
two completely different input images produced **identical** output
token sequences — a hard signal that cross-attention was being
ignored at inference.

Root cause: `MBartForCausalLM.prepare_inputs_for_generation()` does
not propagate `encoder_hidden_states` into the per-step model inputs
during HF's `generate()` loop, so the decoder sees no encoder features
during generation even though they were passed as a kwarg. Same class
of fragility that motivates the optional SDPA monkey-patch.

Fix: replaced the HF `generate()` call inside
`MBartDecoder.generate_greedy` with a hand-rolled greedy loop that
calls `model.forward` each step with `encoder_hidden_states` passed
explicitly; KV cache is preserved so the loop is still O(T) per
sequence. Tests in `tests/` continue to pass.

After the fix and with reasonable repetition control
(`repetition_penalty=1.3`, `no_repeat_ngram_size=6`), greedy outputs
become image-conditioned and structured.

## Reporting your numbers

PRs welcome to fill in this table from your own runs. Please include:

- The hardware you ran on (GPU, VRAM, hours).
- The exact commit hash you ran.
- Anything you changed from the defaults in `configs/base.yaml`.
