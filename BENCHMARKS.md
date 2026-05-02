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

Long-form eval on the same held-out PDFA shard, 100 batches, greedy
decode with the diagnostic `repetition_penalty=1.05` from
`make_val_decode_fn`. Run with:

```
python scripts/eval_pdfa_holdout.py \
  --ckpt checkpoints/stage3/ckpt_best.pt \
  --val-shard data/raw/pdfa/pdfa-eng-train-0119.tar \
  --spm data/processed/vocab/sp_en_16k.model \
  --max-batches 100
```

| Metric | Value |
|---|---|
| Decoded batches | 100 |
| Empty hypotheses | **100 / 100 (100 %)** |
| CER | 1.0000 |
| WER | 1.0000 |
| word-F1 | 0.0000 |
| Wall | 70 s |

**Honest read.** Train loss reached 3.72 (text+loc combined) and val
loss bottomed at 4.32 in stage-2 / 4.46 in stage-3, but every single
greedy hypothesis on the hold-out shard collapses to the empty string.
The teacher-forced loss says the model has learned the joint
distribution; greedy decode under the current generation config
terminates at step 1 every time. Likely culprits to investigate
before re-running:

- HF generation-config interaction with the spatial/special-token
  vocabulary (e.g. EOS being assigned to a different id than the one
  passed to `generate(eos_id=...)`).
- `min_new_tokens=0` in the diagnostic decode allowing immediate EOS.
- A residual pad/eos id collision in the tokenizer or generation kwargs
  surviving the earlier guard (covered by `tests/test_tokenizer.py`
  but worth re-asserting against the trained model's first-token
  argmax).

The hold-out CER row in `BENCHMARKS.md` is intentionally left at the
honest value rather than hidden -- generation must be debugged before
finetune numbers will be meaningful.

## Reporting your numbers

PRs welcome to fill in this table from your own runs. Please include:

- The hardware you ran on (GPU, VRAM, hours).
- The exact commit hash you ran.
- Anything you changed from the defaults in `configs/base.yaml`.
