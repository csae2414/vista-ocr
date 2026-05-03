# SROIE 2019 finetune chain

Reaches the BENCHMARKS.md SROIE row. Needs a pretrained checkpoint
(typically the output of `scripts/pretrain_chain.sh`) and the SROIE
flat layout produced by `scripts/datasets/setup_sroie.sh`.

## One-shot

```bash
# 1. Set up the data once (Kaggle "SROIE datasetv2" by Urban Knupleš).
./scripts/datasets/setup_sroie.sh /path/to/SROIE2019

# 2. Run the finetune + eval chain (defaults match the paper-style
#    short, low-LR finetune profile).
./scripts/benchmarks/sroie/chain.sh
```

The chain reads `checkpoints/stage3/ckpt_best.pt`, finetunes for 5K
steps on SROIE train (with val on test for ckpt_best selection), then
runs `eval.py` against the resulting `ckpt_best.pt` on the full SROIE
test split. The result lands in `logs/eval_sroie.json`; paste the
`word_f1` value into the SROIE row of BENCHMARKS.md.

Tunables are env vars; see the header of `chain.sh`. The most common
ones are `STEPS`, `LR`, `PAGE_PRESET`, and `SROIE_ROOT`.

## Files

* `run.py` -- finetune entry. Loads `--init-from`, trains for
  `--steps`, ckpt_best on val_word_f1 against SROIE test.
* `eval.py` -- decodes every test doc with greedy + word-set P/R/F1.
* `chain.sh` -- env-var wrapper.
