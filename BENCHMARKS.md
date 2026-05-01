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

## Calibration progress (PDFA, no finetune)

End-of-stage validation loss on a held-out PDFA shard. Dropping is good.

| Stage | Steps | Train loss (last 50) | Val loss |
|---|---|---|---|
| Stage-1 calibration | 20K | _TBD_ | _TBD_ |
| Stage-2 multimodal  | 80K | _TBD_ | _TBD_ |
| Stage-3 multitask   | 70K | _TBD_ | _TBD_ |

## Reporting your numbers

PRs welcome to fill in this table from your own runs. Please include:
- The hardware you ran on (GPU, VRAM, hours)
- The exact commit hash you ran
- Anything you changed from the defaults in `configs/base.yaml`
