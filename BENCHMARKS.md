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

> **Split note (2026-05-03 onward).** Two PDFA shards are now reserved at
> the dataset boundary (see `src/vista_ocr/data/split.py`):
> shard **0118** is the val shard used during training for `ckpt_best`
> selection; shard **0119** is the test shard, touched only by
> `scripts/eval_run.sh`. The numbers below were generated against shard
> 0119 *before* the split was locked, so for them shard 0119 was the
> training-time val shard -- **not** a strict held-out test set. Rows
> generated after this date are strict held-out.

Long-form eval on the locked PDFA test shard (0119), 100 batches. Run with:

```
./scripts/eval_run.sh checkpoints/stage3/ckpt_best.pt logs/eval.json
# or, equivalently:
python scripts/eval_pdfa_holdout.py \
  --ckpt checkpoints/stage3/ckpt_best.pt \
  --val-shard data/raw/pdfa/pdfa-eng-train-0119.tar \
  --spm data/processed/vocab/sp_en_16k.model \
  --max-batches 100 \
  --repetition-penalty 1.3 --no-repeat-ngram-size 6 \
  --max-new-tokens 512
```

| Metric | max_new_tokens=512 | max_new_tokens=2048 |
|---|---|---|
| Decoded batches | 100 | 100 |
| Empty hypotheses | 7 / 100 | 6 / 100 |
| CER | **0.8414** | 0.9074 |
| WER | 1.0218 | 1.1516 |
| word-F1 | 0.1562 | 0.1451 |
| Wall | 47 s | 85 s |

Raising the per-sample cap from 512 to 2 048 tokens **made every
metric worse**. This rules out truncation as the bottleneck: the model
emits a few real OCR tokens early, then drifts into a learned PDFA
text prior; the longer cap just lets it emit more hallucinated tokens.
The 512-row is the honest baseline.

**Diagnosis.** The model is image-conditioned (verified — different
images produce different outputs) but the conditioning is *weak*. With
a random-init decoder on a 170 K-step PDFA-only training budget, the
language-model prior dominates whenever the visual signal is
ambiguous. The likely highest-leverage fix is initialising the
decoder body from `facebook/mbart-large-50` instead of random init
(the codebase already has `_copy_body_weights` plumbing for this);
the paper itself indicates an mBART decoder backbone. Other reasonable
follow-ups: more pretraining steps (typical document-OCR convergence
is at 300 K-1 M steps), folding the already-downloaded IDL data into
a mixed pretraining stream, finetune on the target dataset.

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

## Diagnostic ablation table (planned)

To extract per-knob value before booking expensive GPU time, the next
PDFA hold-out evaluation runs a staged ablation on the 3090. Each row
fixes everything except the named knob; baseline = the 2026-05-02 run
documented above.

| Run | Decoder init | Eff batch | Data | Aug | CER | WER | word-F1 |
|---|---|---|---|---|---|---|---|
| Baseline | random | 1 | PDFA-only | off | 0.8414 | 1.0218 | 0.1562 |
| A | **donut** | 1 | PDFA-only | off | _TBD_ | _TBD_ | _TBD_ |
| B | donut | **8** | PDFA-only | off | _TBD_ | _TBD_ | _TBD_ |
| C | donut | 8 | **paper-mix** | **on** | _TBD_ | _TBD_ | _TBD_ |

Decision threshold: word-F1 from Run A must at least double the
baseline (0.156 → 0.30) for the Donut decoder transfer to be
considered effective. If A passes, run B; if B helps further, run C.
Otherwise abort and investigate before any L40s/A100 spend.

Each run uses identical eval flags via `scripts/eval_run.sh
<ckpt-path>`; the only thing that changes between rows is the input
checkpoint.

## Reporting your numbers

PRs welcome to fill in this table from your own runs. Please include:

- The hardware you ran on (GPU, VRAM, hours).
- The exact commit hash you ran.
- Anything you changed from the defaults in `configs/base.yaml`.
