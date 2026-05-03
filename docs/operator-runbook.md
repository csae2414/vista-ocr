# Operator runbook

End-to-end recipe for taking a fresh box, downloading data, training
the chain, and getting a PDFA hold-out number into BENCHMARKS.md.

## 1. Environment

```bash
# CUDA box (CUDA 12.1 + RTX 3090 / L40s / A100):
conda env create -f environment-cuda.yml
conda activate vista-ocr
pip install -e .
pytest -q                   # ~90 seconds, expects ~280 tests passing
```

## 2. Data

```bash
# Default 12 shards (~10 GB):
./scripts/setup_data.sh
# Or more for paper-comparable training:
NUM_SHARDS=120 ./scripts/setup_data.sh
# Optional: noisier real corpus (paper uses both):
python scripts/download_idl.py --num-shards 12
```

`setup_data.sh` also runs `bootstrap_tokenizer.py` to produce the
default SPM model at `data/processed/vocab/sp_en_16k.model`.

## 3. Pretraining chain

```bash
# Recommended defaults (Donut decoder init, effective batch 8,
# augmentation on). All knobs are env-overridable; see the script header.
screen -S train -d -m ./scripts/pretrain_supervised.sh
screen -r train       # to attach
```

Wall-clock estimate on RTX 3090: ~7 hours for 20K + 80K + 70K steps.
The supervisor restarts the chain on any non-zero exit (defaults: 20
attempts, 30 s back-off). The chain auto-resumes from the latest
checkpoint per stage.

To run the previous (paper-default-deviating) configuration:

```bash
INIT_DECODER_FROM=random GRAD_ACCUM_STEPS=1 AUGMENT=0 \
  ./scripts/pretrain_chain.sh 2>&1 | tee logs/pretrain_chain.log
```

### Smoke gate before any long run

```bash
PAGE_PRESET=large GRAD_ACCUM_STEPS=16 SDPA=1 GRAD_CKPT=0 \
  ./scripts/smoke_chain.sh
```

Runs 200 steps of each stage (~3-5 min on L40S). Catches loader,
stage-transition, augment-bbox, and env-wiring bugs before committing
to a multi-day run. Side-effects land in `checkpoints-smoke/`, the
real training's `checkpoints/` is untouched.

### Pre-rendered sample cache (skip pypdfium2 in the hot loop)

For long runs at fixed resolution, pre-render once and read from cache:

```bash
# One-time render (~1-2 h on L40S for 119 PDFA shards at 1400x1050).
python scripts/cache_dataset.py --source pdfa \
  --shards data/raw/pdfa/pdfa-eng-train-*.tar \
  --out-dir /tmp/vista-ocr-data/cache/pdfa-large \
  --page-h 1050 --page-w 1400

python scripts/cache_dataset.py --source idl \
  --shards data/raw/idl/idl-train-*.tar \
  --out-dir /tmp/vista-ocr-data/cache/idl-large \
  --page-h 1050 --page-w 1400
```

The cache is **resolution-bound**: rebuilding is required if you
change `--page-h`, `--page-w`, `--dpi`, or `--score-threshold`.
Re-running the script on a partial cache resumes; safe to kill any
time. Operators on a tight disk budget can point `--out-dir` at NFS.

### Mid-run health checks (before the run is over)

For a multi-day run, plan to look at the metrics three times. Each
check is ~5 min.

1. **+2 hours** (mid-stage 1): step rate matches projection (~0.7-1.2
   s/step on L40S). If much slower, kill and investigate.
2. **End of stage 1**: compare val_loss vs. baseline at same step
   count. Donut init should beat random init by 0.1-0.3 nat.
3. **Stage 2 step 5K**: val_loss + decode-empty rate are early
   signals for whether the data-mix or compile changes are helping.
   If regressed, kill before stage 3 burns more compute.

The cheap watcher (Monitor over `tail -F`) gives you these
automatically; you only need to act on the numbers.

### L40S / 48 GB+ box (extended schedule + speed knobs)

```bash
PAGE_PRESET=large \
GRAD_ACCUM_STEPS=16 \
SDPA=1 \
GRAD_CKPT=0 \
INIT_DECODER_FROM=donut AUGMENT=0 \
STAGE1_STEPS=50000 STAGE2_STEPS=200000 STAGE3_STEPS=100000 \
screen -S train -d -m ./scripts/pretrain_supervised.sh
```

`SDPA=1` runs the C3 ship-gate first (forward / backward / KV-cache /
beam log-prob / autocast bf16 paper shape), aborts on failure, then
applies the patch and writes the PASS line to
`<out>/sdpa_manifest.txt`. Measured on L40S: ~10× kernel speedup, −17 %
peak memory.

`GRAD_CKPT=0` disables encoder gradient checkpointing — frees the
encoder forward from recompute on the backward pass. Safe at large
preset on 48 GB. Combined with SDPA, expect ~2× per-step speedup
vs. the L40S default.

Auto-launch when downloads finish (separate screen):

```bash
PDFA_TARGET=120 IDL_TARGET=12 \
PAGE_PRESET=large GRAD_ACCUM_STEPS=16 SDPA=1 GRAD_CKPT=0 \
INIT_DECODER_FROM=donut AUGMENT=0 \
STAGE1_STEPS=50000 STAGE2_STEPS=200000 STAGE3_STEPS=100000 \
tmux new-session -d -s launcher ./scripts/launch_when_ready.sh
```

## 4. Hold-out evaluation

The PDFA shards are split into three roles
(see ``src/vista_ocr/data/split.py``):

| Role | Shard basename | Touched by |
|---|---|---|
| Train | every other shard (0000..0117) | stage1/2/3_run.py |
| Val   | ``pdfa-eng-train-0118.tar`` | training (ckpt_best selection) |
| Test  | ``pdfa-eng-train-0119.tar`` | ``scripts/eval_run.sh`` only |

Stage scripts call ``assert_not_test_shard`` on every shard argument
and refuse to start if the test shard sneaks into ``--val-shard`` or
``--train-shards``. Run the held-out eval with:

```bash
# Reproducible eval -- fixed flags, deterministic seed, locked test shard.
./scripts/eval_run.sh checkpoints/stage3/ckpt_best.pt logs/run_A.json
./scripts/eval_run.sh checkpoints/stage3/ckpt_final.pt logs/run_A_final.json
```

Result JSON has CER, WER, word-F1, empty-fraction, decoded-batch
count, and the underlying ckpt step. Paste these into the
"Diagnostic ablation table" in `BENCHMARKS.md`.

## 5. Live monitoring (optional)

A side-process tailer parses the supervisor log into JSONL +
TensorBoard scalars without touching the running training. Start it
in a separate `screen` so it stays up across train restarts:

```bash
screen -S tailer -d -m bash -c \
  "source ~/miniconda3/etc/profile.d/conda.sh && conda activate vista-ocr && \
   python scripts/log_tailer.py"
screen -S tb -d -m bash -c \
  "source ~/miniconda3/etc/profile.d/conda.sh && conda activate vista-ocr && \
   tensorboard --logdir logs/tb --bind_all --port 6006"
```

TensorBoard is then reachable at `http://<vm-ip>:6006`. The tailer
emits structured records to `logs/metrics.jsonl` for downstream
grepping.

## 6. Comparing checkpoints

For an A/B comparison: run the chain twice with different
`INIT_DECODER_FROM` (or any other env var); evaluate each with
`scripts/eval_run.sh`; diff the resulting JSONs. The diagnostic
ablation table in `BENCHMARKS.md` is the canonical home for these
numbers.

## 7. SDPA fast attention (optional)

```bash
# One-time check on this box + transformers version:
python -m vista_ocr.models.sdpa_patch --check
# Per-stage opt-in (writes <out>/sdpa_manifest.txt for traceability):
python scripts/stage2_run.py --sdpa ...
```

The patch is opt-in and gated by the ship-test (forward, backward,
KV-cache, beam log-prob, autocast bf16 at the paper attention
shape). Default-off so all numbers are paper-comparable.
