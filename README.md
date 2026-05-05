# vista-ocr

> **Unofficial / third-party reimplementation.** This repository is not
> authored by, affiliated with, or endorsed by the original VISTA-OCR
> authors. The original paper has no public reference implementation;
> this is a clean-room PyTorch port based on the published description.

A from-scratch PyTorch reimplementation of **VISTA-OCR** (Hamdi, Tamasna,
Boisson, Paquet — *VISTA-OCR: Towards generative and interactive end to
end OCR models*, [arXiv:2504.03621](https://arxiv.org/html/2504.03621v1),
April 2025).

No official code has been released by the authors. This repository follows
the paper as faithfully as possible, with each gap (hyperparameters the
paper does not specify) backed by sister-model evidence noted in commit
messages and module docstrings.

> **Status.** Architecture and full training pipeline implemented and
> verified end-to-end on a single RTX 3090 / L40S. Stage-1 calibration
> runs on real PDFA pages; checkpoint + resume + validation work; all
> dataset loaders (PDFA, IDL, IAM, MAURDOR, SROIE) tested. The chain
> supports `DATA_MIX={pdfa | pdfa+idl | pdfa+synth | pdfa+idl+synth}`,
> with a license-clean handwritten synth generator (Phase J) and an
> optional unfrozen OCR-only stage 1b (Phase I) between the frozen
> stage 1 and the multimodal stage 2. Long-form pretraining
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
- **Curriculum** (calibration -> optional unfrozen-text-only ->
  multimodal -> multitask) with checkpoint + auto-resume, periodic
  validation, and ckpt_best on val improvement. Stage 1b
  (`stage1b_run.py`, Phase I) is paper §3.6.1's unfrozen text-only
  bridge between the frozen stage 1 and the multimodal stage 2;
  off by default (`STAGE1B_STEPS=0`), opt in via env.
- **Data mix** (`DATA_MIX` env on `pretrain_chain.sh`, Phase H + J)
  selects one of `pdfa | pdfa+idl | pdfa+synth | pdfa+idl+synth`.
  IDL is paper-comparable (60% real / 40% PDFA at default
  `IDL_WEIGHT=0.6`); synth is the license-clean handwritten
  generator (Phase J, opt-in). Stage 1 + 1b stay PDFA-only; mixing
  applies to stages 2 + 3.
- **Synthetic handwritten data**
  (`vista_ocr.data.synth.handwritten.HandwrittenLineSynth`,
  Phase J) generates pages from a local text corpus + bundled
  OFL/Apache TTF font set with mask-derived bboxes. License-clean
  approximation, **NOT a paper-equivalent reproduction** of
  synthetic IAM/RIMES (per-writer variation not modeled).
  See `docs/operator-runbook.md` §3.1 for the workflow.
- **Speed**: bf16 autocast, encoder gradient checkpointing, KV-cache
  decoding, multi-worker DataLoader. Optional opt-in
  `MBartAttention -> SDPA` monkey-patch with a CI ship-gate (forward,
  backward, KV-cache, beam log-prob, autocast bf16 at the paper shape;
  `python -m vista_ocr.models.sdpa_patch --check`).
- **Eval**: CER, WER, word-exact F1, Wolf & Jolion DetEval (paper
  ref [37]), Area-F1, AP@IoU{0.5, 0.6, 0.7, 0.8}.
- **VRAM-aware page resolution** via `--page-preset {tiny, small,
  medium, large, paper, auto}` on the stage scripts; `auto` queries
  CUDA VRAM and picks a sensible canvas (medium = 1100×850 fits the
  24 GB 3090; large = 1400×1050 is the L40S default; paper =
  2200×1700 matches paper §3.5).
- **Decoder initialisation from Donut** (`--init-decoder-from donut`)
  per VISTA-OCR §3.2. Loads the BART text-decoder body from
  `naver-clova-ix/donut-base`; vocab-shaped tensors are skipped
  (different tokenizer) and re-trained from random init.
- **Speed knobs** for 48 GB+ cards: `SDPA=1` enables the optional
  10× attention kernel (ship-gate runs first), `GRAD_CKPT=0`
  disables encoder gradient checkpointing for a further ~30-50 %
  per-step speedup. Both are env-tunable on `pretrain_chain.sh`.
- **Early stopping** (`EARLY_STOP=1`) with EMA-smoothed patience +
  spike detection. Resume-safe (state persists into the checkpoint)
  and structured `EARLY_STOP:` log line for operator parsing. Saves
  wall-clock when training plateaus before the configured step
  budget; per-stage defaults differ (calibration > multimodal).
- **On-disk sample cache** (`scripts/cache_dataset.py`) for fixed-
  resolution training runs — pre-renders PDFA / IDL once, then the
  hot loop is pure I/O. Geometry-bound manifest, atomic write,
  resume on partial render.
- **Dataset adapters** (`scripts/datasets/`) flatten benchmark-
  specific layouts into our loader's expected form. SROIE shipped
  (`setup_sroie.sh`); synthetic-corpus acquisition shipped
  (`setup_synth_corpus.sh`, Phase J); IAM is licence-restricted at
  acquisition time but the loader (`vista_ocr.data.iam`) is in
  place; MAURDOR / PageXML loaders are the planned slots.
- **Locked PDFA val/test split**: shard `0118` is the val shard
  (used for `ckpt_best` selection); shard `0119` is the test shard,
  touched only by `scripts/eval_run.sh`. Training scripts call
  `assert_not_test_shard` to refuse the test shard.
- **523 unit tests** (4 skipped on environments lacking DejaVu test
  font), Sphinx API docs, MIT licensed.

## Quick start

The fastest path uses three helper scripts. Each is idempotent and safe
to rerun.

### 1. Install

`vista-ocr` is a regular Python package; you can install with plain
pip and pick the right torch wheel via `--extra-index-url`. No conda
required.

```bash
# CPU only (Linux / macOS, e.g. CI)
pip install -e ".[cpu]"        --extra-index-url https://download.pytorch.org/whl/cpu

# GPU box, CUDA 12.1 (RTX 3090, L40S)
pip install -e ".[gpu-cu121]"  --extra-index-url https://download.pytorch.org/whl/cu121

# GPU box, CUDA 12.4
pip install -e ".[gpu-cu124]"  --extra-index-url https://download.pytorch.org/whl/cu124
```

After install, the `vista-ocr` console script is on your PATH; verify:

```bash
vista-ocr --version
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
pytest -q                         # ~2 minutes
```

If you prefer conda, the legacy `environment.yml` / `environment-cuda.yml`
files still work — `pip install -e ".[cpu]"` (or `gpu-cu121`) inside the
activated env adds the entry point and pins the runtime deps.

### CLI

The `vista-ocr` console script exposes the Python verbs; the legacy
`scripts/*.py` invocations and `scripts/*.sh` chain wrappers continue
to work as before (operators in mid-flight runs are unaffected).

| Verb | Purpose |
|---|---|
| `vista-ocr stage {1\|1b\|2\|3}` | Pretraining stage (mirrors `scripts/stage{N}_run.py`). Stage 1b is the optional Phase-I unfrozen text-only bridge. |
| `vista-ocr eval --manifest <jsonl>` | Evaluate a checkpoint against a JSONL manifest; writes a JSON sidecar. Computes recognition (CER/WER/word-F1) always; detection (DetEval/Area-F1/AP@IoU) when manifest has `bboxes`; AP@CER (region-OCR) with `--cer-ap-thresholds`. Paper-faithful SROIE detection rows via `--bbox-expand-px 2`. |
| `vista-ocr finetune --train-manifest X --val-manifest Y --init-from K` | Generic manifest-driven finetune. |
| `vista-ocr infer --folder <root>` | Decode every image in a folder; output JSONL with a `_meta` header (ckpt path/step, timestamp). |
| `vista-ocr cache` | Pre-render dataset to a geometry-bound cache (mirrors `scripts/cache_dataset.py`). |

#### JSONL manifest

`finetune` and `eval` consume a JSONL manifest where each line is
one document (see `vista_ocr.data.manifest`):

```jsonl
{"image": "img/001.jpg", "ref": "Hello world"}
{"image": "img/002.jpg", "ref": "Foo bar", "bboxes": [[10,20,100,50,"Foo"]]}
```

Required: `image` (relative to the manifest's dir, or absolute), `ref`.
Optional: `bboxes` (`[[x1,y1,x2,y2,text], ...]`), `task` (default
`ocr_layout`), `query_text` / `query_bbox`, `version` (default 1;
unknown versions are a hard error).

Per-benchmark adapters in `scripts/datasets/` produce manifests, e.g.:

```bash
./scripts/datasets/setup_sroie.sh /path/to/SROIE2019 \
    --emit-manifests data/sroie/manifests
```

### 2. Pull data + train tokenizer (≈ 10 GB, one-shot)

```bash
./scripts/setup_data.sh                # 12 PDFA shards from HuggingFace
# or: NUM_SHARDS=24 ./scripts/setup_data.sh   # 24 shards (~20 GB)
```

This calls `scripts/download_pdfa.py` (HuggingFace `pixparse/pdfa-eng-wds`,
shards 0000..0011 by default) and `scripts/bootstrap_tokenizer.py`
(WikiText-2 SentencePiece). Outputs end up in `data/raw/pdfa/` and
`data/processed/vocab/`.

For a richer mixed printed + handwritten corpus, also pull
[HierText](https://huggingface.co/datasets/google-research-datasets/hiertext)
(~12 GB, CC-BY-4.0):

```bash
python scripts/download_hiertext.py
```

### 3. Pretrain (stage-1 -> [stage-1b] -> stage-2 -> stage-3, unattended)

```bash
tmux new -s pretrain
./scripts/pretrain_chain.sh 2>&1 | tee logs/pretrain_chain.log
# detach: Ctrl-b d ; reattach: tmux attach -t pretrain
```

The chain runs the configured stages back-to-back, uses the locked
val shard `pdfa-eng-train-0118.tar`, writes `checkpoints/stage{1,1b,2,3}/`,
and **resumes automatically** if killed mid-run (just rerun the same
command). For supervised long runs use `pretrain_supervised.sh`,
which wraps the chain in a restart-on-failure supervisor.

Common env knobs (full list in the script header):

| Env | Default | Effect |
|---|---|---|
| `DATA_MIX` | `pdfa` | One of `pdfa`, `pdfa+idl`, `pdfa+synth`, `pdfa+idl+synth`. Stages 2+3 only; stage 1 + 1b stay PDFA-only. |
| `IDL_WEIGHT` | `0.6` | IDL fraction of the mix when `DATA_MIX` includes `idl`. |
| `SYNTH_WEIGHT` | `0.2` | Synth fraction when `DATA_MIX` includes `synth`. Hard-fails if `IDL_WEIGHT + SYNTH_WEIGHT >= 1`. |
| `STAGE1B_STEPS` | `0` | `0` skips stage 1b (legacy chain). Set ≥ 10000 for paper-faithful curriculum. |
| `PAGE_PRESET` | `medium` | `tiny\|small\|medium\|large\|paper\|auto`; `auto` queries CUDA VRAM. |
| `EARLY_STOP` | `0` | `1` enables EMA-smoothed plateau abort with auto-cap of `STAGE_N_STEPS`. |

Each stage script also accepts `--page-preset` and `--sdpa` (the
SDPA monkey-patch with ship-gate; manifest written to
`<out>/sdpa_manifest.txt` for paper-comparison reproducibility).

Expect ≈ 0.13 s / step on a single RTX 3090 -> roughly **6 hours**
total at the default 20K + 80K + 70K steps with `medium` preset.
At `large` preset on an L40S, expect ~2 s / step.

### 4. Finetune + benchmark

Once IAM / MAURDOR / SROIE are downloaded under their licences into
`data/raw/<dataset>/` (see each loader docstring for the layout):

```bash
./scripts/finetune_chain.sh
# or: DATASETS="sroie" ./scripts/finetune_chain.sh    # subset
```

This runs `scripts/finetune_eval.py` per dataset with the paper's batch
sizes (RIMES=4, IAM=6, SROIE=2 — paper Section 4.2) and writes one log
per dataset under `logs/`. Compare the printed metrics to
[`BENCHMARKS.md`](BENCHMARKS.md) which has the paper targets.

### 5. Verify against the paper

```bash
grep -E "F1|WER|Area" logs/finetune_*.log
# fill the result into BENCHMARKS.md and open a PR if you reproduce
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
| 1 -- calibration | `stage1_run.py` | frozen | OCR (text only) | 3e-4 | 20K |
| 1b -- bridge (Phase I, opt-in) | `stage1b_run.py` | unfrozen | OCR (text only, λ=1.0) | 5e-5 | 0 (default skip) / 10K |
| 2 -- multimodal | `stage2_run.py` | unfrozen | OCR + layout (interleaved) | 5e-5 | 80K |
| 3 -- multitask | `stage3_run.py` | unfrozen | OCR / OCR+layout / region_OCR / find_it | 3e-5 | 70K |
| FT -- per dataset | `finetune_eval.py` | unfrozen | dataset-specific | 1e-5 | 5K-10K |

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
├── configs/             YAML configs (base + per-stage + per-finetune)
├── docs/                Sphinx API docs (run: cd docs && make html)
├── scripts/
│   ├── setup_data.sh              Helper: download PDFA + train tokenizer
│   ├── pretrain_chain.sh          Helper: stage 1 -> [1b] -> 2 -> 3
│   ├── pretrain_supervised.sh     Restart-on-failure supervisor wrapper
│   ├── finetune_chain.sh          Helper: per-dataset finetune + eval
│   ├── eval_run.sh                Reproducible eval against locked test shard 0119
│   ├── cache_dataset.py           Pre-render PDFA/IDL to a geometry-bound cache
│   ├── download_pdfa.py           Pull N shards from HuggingFace
│   ├── bootstrap_tokenizer.py     Train SPM on WikiText-2
│   ├── stage1_run.py              Stage 1 calibration (frozen decoder)
│   ├── stage1b_run.py             Stage 1b bridge (Phase I; unfrozen text-only)
│   ├── stage2_run.py              Stage 2 multimodal
│   ├── stage3_run.py              Stage 3 multitask
│   ├── finetune_eval.py           Per-dataset finetune + benchmark
│   ├── ablate_lambda.py           λ ∈ {0.3, 0.5, 0.7} sweep
│   ├── ablate_scheme.py           Original / Segmented / Unified ablation
│   ├── decoder_ab.py              12-layer mBART vs 4-layer Donut
│   ├── bench_dataloader.py        DataLoader workers/compile/SDPA bench
│   ├── smoke_test_gpu.py          5-minute end-to-end GPU sanity
│   ├── diag_val_crash.py          CUDA-launch-blocking val diagnostic
│   └── datasets/
│       ├── setup_sroie.sh         Flatten SROIE 2019 into JSONL manifests
│       └── setup_synth_corpus.sh  One-time corpus acquisition for Phase J synth
├── src/vista_ocr/
│   ├── tokenizer/       SentencePiece + spatial grid + 3 schemes
│   ├── models/          DANIEL encoder, mBART/Donut decoder, VistaOCR wrapper
│   ├── ablation/        Ablation harness shared by ablate_*.py
│   ├── data/            Preprocess, collate, multi-worker dataloader,
│   │                    pdfa, idl, iam, maurdor, sroie loaders + synth/
│   │                    (handwritten generator + factory + fonts),
│   │                    BBox value object, PdfaShardReader
│   ├── training/        Combined loss, schedules, callbacks, train loop
│   ├── eval/            CER/WER, Wolf & Jolion DetEval, AP@IoU
│   └── inference/       Greedy/beam generation + output parser
├── tools/
│   ├── audit_fr_coverage.py            Phase J0 #2 tokenizer FR-coverage gate
│   └── synth_target_distributions.py   Phase J0 #3 sanity-band baseline
├── corpora/synth/letterlike.txt        In-tree letter-shaped fixture (Phase J)
└── tests/                              523 unit tests (4 skipped on non-DejaVu hosts)
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
  documented inline in `configs/base.yaml` and the relevant module
  docstrings
- Data augmentations beyond Appendix 0.A.4 → implemented per appendix
  for synth-SROIE, default minimal for others

Empirical findings (decoder A/B between mBART-12 and Donut-4, λ sweep,
speed-knob outcomes) live in the commit history and the relevant
ablation scripts under `scripts/`.

## Reproducibility checklist

| Item | Status |
|---|---|
| Architecture matches paper Section 3 | ✅ |
| Tokenizer / 3 encoding schemes | ✅ |
| Combined loss + prompt masking | ✅ |
| Curriculum (1 -> [1b] -> 2 -> 3) | ✅ |
| All five real dataset loaders | ✅ |
| SynthDOG-bbox + SROIE-synth (paper appendix) | ✅ |
| Phase H: paper-comparable PDFA + IDL data mix | ✅ |
| Phase I: stage 1b unfrozen-OCR-only bridge (paper §3.6.1) | ✅ |
| Phase J: license-clean handwritten synth + chain integration | ✅ |
| End-to-end training on real PDFA + checkpoints | ✅ |
| Eval metrics (CER, WER, F1, Wolf & Jolion, Area-F1, AP@IoU) | ✅ |
| Locked PDFA val (0118) / test (0119) split | ✅ |
| Sphinx API docs | ✅ |
| 523 unit tests passing (4 skipped on non-DejaVu hosts) | ✅ |
| SDPA fast attention (opt-in monkey-patch + ship-gate) | ✅ |
| VRAM-aware page resolution presets (incl. paper @ 2200×1700) | ✅ |
| Long pretraining runs at scale | 🟡 plumbing verified, runtime hours |
| Per-dataset finetune numbers vs paper | 🟡 needs licence-restricted data |
| Synthetic IAM/RIMES paper-equivalent (per-writer variation) | 🟡 license-clean approximation only |
| `torch.compile` win | 🟡 needs fixed-bucket page sizes |

## Hardware notes

- We trained on a single **RTX 3090 (24 GB)**, not the paper's A100 80GB.
  Mandatory consequences: gradient checkpointing always on, micro-batch
  =1 at full page resolution, ~3-4× slower wall-clock vs A100.
- bf16 autocast in eval mode hits PyTorch
  [#132613](https://github.com/pytorch/pytorch/issues/132613) on the
  MBart eager attention path; we run validation in fp32 as a workaround.
- `MBartForCausalLM` SDPA / FlashAttention-2 is unsupported in
  `transformers==4.44.2`
  ([HF #28005](https://github.com/huggingface/transformers/issues/28005)).
  Workaround: an opt-in monkey-patch in
  `vista_ocr.models.sdpa_patch` with a ship-gate (forward / backward /
  KV-cache / beam log-prob / autocast bf16 at the paper shape).
  Enable per-stage with `--sdpa`; the gate writes a manifest to
  `<out>/sdpa_manifest.txt`. Disable with `VISTA_NO_SDPA=1`; skip the
  gate (after operator verification) with `VISTA_SDPA_SKIP_CHECK=1`.

## References

- VISTA-OCR paper: [arXiv:2504.03621](https://arxiv.org/html/2504.03621v1)
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

## Citing the paper

This repository is not the artefact to cite. If this implementation
helps your work, please cite the original paper:

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
