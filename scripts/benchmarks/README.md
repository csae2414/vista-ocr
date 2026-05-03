# `scripts/benchmarks/`

Per-benchmark finetune + eval entry points. Mirrors the `scripts/datasets/`
folder convention: keep benchmark-specific glue out of the top-level
`scripts/` namespace, which is reserved for generic + pretraining tools.

Each subdirectory is one benchmark (one row in `BENCHMARKS.md`):

| Subdir | Benchmark | Status |
|---|---|---|
| `sroie/` | SROIE 2019 word-F1 | scaffolded |
| `iam/` | IAM WER | TBD (registration required) |
| `maurdor/` | MAURDOR-EN Area-F1 | not reproducible (paid corpus) |

## Convention

Each benchmark folder ships:

* `run.py` -- the finetune entry point (loads a pretrained ckpt, trains
  on the benchmark train split, ckpt_best on val_word_f1).
* `eval.py` -- the post-finetune full-test-split evaluator (writes a
  JSON sidecar suitable for pasting into BENCHMARKS.md).
* `chain.sh` -- the env-var-driven wrapper that orchestrates `run.py`
  then `eval.py` so a single command produces a benchmark row.

The pretraining chain (`scripts/pretrain_chain.sh`) is canonical and
benchmark-independent; finetuning + benchmark eval lives here.
