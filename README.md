# vista-ocr

A from-scratch PyTorch reimplementation of **VISTA-OCR** (Hamdi, Tamasna,
Boisson, Paquet — *VISTA-OCR: Towards generative and interactive end to
end OCR models*, [arXiv:2504.03621](https://arxiv.org/abs/2504.03621),
April 2025).

No official code has been released by the authors. This repository follows
the paper as faithfully as possible, with every engineering decision
documented in [`PLAN.md`](PLAN.md) and [`PLAN_VM.md`](PLAN_VM.md).

> **Status.** Architecture and full training pipeline implemented and
> verified end-to-end on a single RTX 3090. Stage-1 calibration runs on
> real PDFA pages; checkpoint + resume + validation work; all dataset
> loaders (PDFA, IDL, IAM, MAURDOR, SROIE) tested. Long-form pretraining
> (Stage-2 80K / Stage-3 70K steps) is one shell command away. Headline
> finetune numbers are *not* yet reproduced — that needs the long
> pretraining runs plus the licence-restricted IAM/MAURDOR/SROIE
> datasets.

## Highlights

- **Encoder** transcribed exactly from
  [DANIEL](https://github.com/Shulk97/daniel) (`FCN_Encoder_Widther`,
  same lab as VISTA-OCR, paper ref [14]) with factorized 2-D positional
  embedding.
- **Decoder** is a 12-layer mBART or 4-layer Donut-style BART, both
  available; A/B benchmarked on the GPU (donut4 wins on
  parameters/throughput at indistinguishable loss).
- **Tokenizer** implements all three serialization schemes from paper
  Table 7 (Original / Segmented / Unified), the literal-text region-OCR
  prompt from Fig. 6 (`Read at x1,y1,x2,y2`), and the `<find_it>` query
  format from Fig. 7.
- **Loss** = λ · L<sub>text</sub> + (1 − λ) · L<sub>loc</sub> with
  prompt-token masking and label smoothing.
- **Three-stage curriculum** (calibration → multimodal → multitask)
  with checkpoint + auto-resume, periodic validation, and ckpt_best on
  val improvement.
- **Speed**: bf16 autocast, encoder gradient checkpointing, KV-cache
  decoding, multi-worker DataLoader. Empirically benchmarked on the GPU
  (PLAN_VM has the numbers).
- **Eval**: CER, WER, word-exact F1, Wolf & Jolion DetEval (paper
  ref [37]), Area-F1, AP@IoU{0.5, 0.6, 0.7, 0.8}.
- **101 unit tests**, Sphinx API docs, MIT licensed.

## Quick start

### 1. Environment

```bash
# CPU dev (Linux/macOS)
conda env create -f environment.yml
conda activate vista-ocr
pip install -e .
pytest -q                         # 101 tests, ~1 minute

# GPU box (CUDA 12.1)
conda env create -f environment-cuda.yml
conda activate vista-ocr
pip install -e .
```

### 2. Train a SentencePiece tokenizer

A small WikiText-2 corpus is enough to bootstrap the spatial-token grid
+ subword vocab. Once you have PDFA shards downloaded, retrain on the
real corpus for better subword splits.

```bash
python scripts/bootstrap_tokenizer.py
# -> data/processed/vocab/sp_en_16k.model
```

### 3. Pull PDFA shards

```bash
python -c "
from huggingface_hub import hf_hub_download
for i in range(12):
    hf_hub_download(
        'pixparse/pdfa-eng-wds',
        filename=f'pdfa-eng-train-{i:04d}.tar',
        repo_type='dataset',
        local_dir='data/raw/pdfa',
    )
"
# 12 shards ≈ 10 GB
```

### 4. Stage-1 calibration

Frozen-decoder warm-up at LR 3e-4. Stops if loss diverges; auto-resumes
if killed and restarted with the same flags.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/stage1_run.py \
  --train-shards data/raw/pdfa/pdfa-eng-train-000{0,1,2}.tar \
  --val-shard    data/raw/pdfa/pdfa-eng-train-0003.tar \
  --spm          data/processed/vocab/sp_en_16k.model \
  --out          checkpoints/stage1 \
  --steps 20000 --val-every 500 --ckpt-every 500
```

Expect ≈ 0.13 s / step on an RTX 3090.

### 5. Stage-2 + Stage-3 chain

The ready-made `pretrain_chain.sh` (generated for the GPU VM) runs
multimodal then multitask pretraining. Run under tmux so SSH disconnect
doesn't kill it.

```bash
tmux new -s pretrain
./pretrain_chain.sh 2>&1 | tee logs/pretrain_chain.log
# detach: Ctrl-b d ; reattach: tmux attach -t pretrain
```

### 6. Finetune + benchmark

Once IAM / MAURDOR / SROIE are downloaded under their licences:

```bash
python scripts/finetune_eval.py --dataset sroie  --root data/raw/sroie  \
    --spm data/processed/vocab/sp_en_16k.model                          \
    --checkpoint checkpoints/stage3/ckpt_best.pt --steps 5000

python scripts/finetune_eval.py --dataset iam     --root data/raw/iam     ...
python scripts/finetune_eval.py --dataset maurdor --root data/raw/maurdor ...
```

## Architecture

```
                        ┌───────────────────────┐
   grayscale page  ───▶ │ DANIEL FCN encoder    │
   (e.g. 1100×850)      │ stride (32, 8)        │
                        │ 21.5 M params, d=1024 │
                        └──────────┬────────────┘
                                   │ (B, S, 1024)
                                   ▼
                        ┌───────────────────────┐
   prompt + targets ──▶ │ mBART/Donut decoder   │ ──▶ next-token logits
                        │ 4-12 layers, d=1024   │
                        │ 88-222 M params       │
                        └───────────────────────┘
```

- **Encoder** — `vista_ocr.models.encoder.FCNEncoderWidther`. Six
  `ConvBlock` + four `DSCBlock` stages widening 1→1024 channels with
  stride (32, 8), then a factorized H+W learned positional embedding
  (`h_max=500`, `w_max=1000`).
- **Decoder** — `vista_ocr.models.decoder.MBartDecoder`. The default is
  `donut4` (4-layer BART, ~88 M params) per the empirical A/B on the
  GPU; the paper-literal `mbart12` (12-layer mBART, ~222 M params) is a
  one-line switch.
- **Output sequence** — paper Table 7 "Original" interleaved scheme:
  `<x_left><y_top> w₁ w₂ … wₘ <x_right><y_bottom>` per line, in
  top-left-to-bottom-right reading order, lines concatenated.
- **Region-OCR prompt** — literal text per Fig 6:
  `<task=region_ocr> Read at 66,184,97,186`. Coordinates are
  subword-tokenized digits in the prompt; spatial tokens only appear in
  the *output*.

See `docs/` for the rendered Sphinx API documentation.

## Training pipeline

Three pretraining stages then per-dataset finetune:

| Stage | Script | Decoder | Tasks | LR | Steps |
|---|---|---|---|---|---|
| 1 — calibration | `stage1_run.py` | frozen | OCR (text only) | 3e-4 | 20K + 30K |
| 2 — multimodal | `stage2_run.py` | unfrozen | OCR + layout (interleaved) | 5e-5 | 80K |
| 3 — multitask | `stage3_run.py` | unfrozen | OCR / OCR+layout / region_OCR / find_it | 3e-5 | 70K |
| FT — per dataset | `finetune_eval.py` | unfrozen | dataset-specific | 1e-5 | 5K-10K |

All stages share:

- AdamW with mBART betas `(0.9, 0.98)` and eps `1e-6`
- bf16 autocast on GPU
- Encoder gradient checkpointing
- Linear warm-up + cosine decay
- Multi-worker DataLoader over PDFA WebDataset shards
- Checkpoint + auto-resume (kill anytime, restart with the same flags)
- Held-out val with `ckpt_best.pt` saved on val_loss improvement

## Repository layout

```
vista-ocr/
├── PLAN.md              ┐ Implementation plan + paper-faithfulness audit
├── PLAN_VM.md           ┘ + GPU-VM-specific notes (HP, speed, decisions)
├── configs/             YAML configs (base + per-stage + per-finetune)
├── docs/                Sphinx API docs (run: cd docs && make html)
├── scripts/
│   ├── bootstrap_tokenizer.py     Train SPM on WikiText-2
│   ├── stage1_run.py              Stage-1 calibration
│   ├── stage2_run.py              Stage-2 multimodal
│   ├── stage3_run.py              Stage-3 multitask
│   ├── finetune_eval.py           Per-dataset finetune + benchmark
│   ├── ablate_lambda.py           λ ∈ {0.3, 0.5, 0.7} sweep
│   ├── ablate_scheme.py           Original / Segmented / Unified ablation
│   ├── decoder_ab.py              12-layer mBART vs 4-layer Donut
│   ├── bench_dataloader.py        DataLoader workers/compile/SDPA bench
│   ├── smoke_test_gpu.py          5-minute end-to-end GPU sanity
│   └── diag_val_crash.py          CUDA-launch-blocking val diagnostic
├── src/vista_ocr/
│   ├── tokenizer/       SentencePiece + spatial grid + 3 schemes
│   ├── models/          DANIEL encoder, mBART/Donut decoder, VistaOCR wrapper
│   ├── data/            Preprocess, collate, multi-worker dataloader,
│   │                    pdfa, idl, iam, maurdor, sroie loaders + synth/
│   ├── training/        Combined loss, schedules, callbacks, train loop
│   ├── eval/            CER/WER, Wolf & Jolion DetEval, AP@IoU
│   └── inference/       Greedy/beam generation + output parser
└── tests/               101 unit tests
```

## Paper faithfulness

The paper specifies:

- The architecture family (CNN encoder + mBART decoder, 150M params)
- The output serialization (`<x><y>w...<x><y>`)
- The three encoding schemes (Table 7)
- The three-stage curriculum
- The combined loss formula
- The four task prompts
- The datasets (PDFA, IDL, IAM, RIMES, SROIE, MAURDOR + 4 synthetic)
- The reported targets (SROIE F1=93.95, IAM WER=10.14, MAURDOR
  Area-F1=87.02)
- DetEval = Wolf & Jolion 2006 (ref [37])

The paper does **not** specify:

- Encoder layer counts or channel widths → recovered exactly from DANIEL
- mBART checkpoint or layer count → 12 layers per the paper-literal
  reading; A/B-tested 4-layer Donut variant lands at the paper's stated
  150 M total params and is now the default
- Vocabulary size → 16 k SentencePiece (configurable; documented)
- λ in the loss → 0.5 default with {0.3, 0.5, 0.7} ablation
- LR, warmup, schedule, total steps, weight decay, gradient clipping
  → all defaults sourced from sister models (TrOCR, Donut, mBART) and
  documented in `PLAN_VM.md` "Hyperparameter starting points"
- Data augmentations beyond Appendix 0.A.4 → implemented per appendix
  for synth-SROIE, default minimal for others

`PLAN.md` §12.5 lists every gap with our default and a faithfulness
score; `PLAN_VM.md` extends with empirical findings (decoder A/B,
λ sweep, speed knob outcomes) from the GPU.

## Reproducibility checklist

| Item | Status |
|---|---|
| Architecture matches paper Section 3 | ✅ |
| Tokenizer / 3 encoding schemes | ✅ |
| Combined loss + prompt masking | ✅ |
| Three-stage curriculum | ✅ |
| All five real dataset loaders | ✅ |
| SynthDOG-bbox + SROIE-synth (paper appendix) | ✅ |
| End-to-end training on real PDFA + checkpoints | ✅ |
| Eval metrics (CER, WER, F1, Wolf & Jolion, Area-F1, AP@IoU) | ✅ |
| Stage-1 → Stage-2 → Stage-3 chain script | ✅ |
| Sphinx API docs | ✅ |
| 101 / 101 unit tests passing | ✅ |
| Long pretraining runs at scale | 🟡 plumbing verified, runtime hours |
| Per-dataset finetune numbers vs paper | 🟡 needs licence-restricted data |
| SDPA / FlashAttention-2 fast attention path | 🟡 blocked by HF #28005 |
| `torch.compile` win | 🟡 needs fixed-bucket page sizes |

## Hardware notes

- We trained on a single **RTX 3090 (24 GB)**, not the paper's A100 80GB.
  Mandatory consequences: gradient checkpointing always on, micro-batch
  =1 at full page resolution, ~3-4× slower wall-clock vs A100.
- All speed and reliability findings (working and not) are recorded in
  `PLAN_VM.md`. Notable: bf16 autocast in eval mode hits PyTorch
  [#132613](https://github.com/pytorch/pytorch/issues/132613); we run
  validation in fp32 as a workaround.

## References

- VISTA-OCR paper: [arXiv:2504.03621](https://arxiv.org/abs/2504.03621)
  (Hamdi et al., 2025)
- DANIEL encoder reference: Constum, Tranouez, Paquet, *DANIEL: A Fast
  Document Attention Network…*, IJDAR 2025,
  [arXiv:2407.09103](https://arxiv.org/abs/2407.09103),
  [github.com/Shulk97/daniel](https://github.com/Shulk97/daniel)
- DetEval protocol: Wolf, Jolion, *Object Count/Area Graphs for the
  Evaluation of Object Detection and Segmentation Algorithms*, IJDAR
  2006

## License

MIT — see [`LICENSE`](LICENSE).

## Citing

If this implementation helps your work, please cite the paper:

```bibtex
@article{hamdi2025vistaocr,
  title  = {VISTA-OCR: Towards generative and interactive end to end OCR models},
  author = {Hamdi, Laziz and Tamasna, Amine and Boisson, Pascal and Paquet, Thierry},
  journal= {arXiv preprint arXiv:2504.03621},
  year   = {2025},
}
```

---

Built without official code. Pull requests welcome — especially around
the licence-restricted finetune evaluations and the long pretraining
artefacts.
