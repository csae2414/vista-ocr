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

## 4. Hold-out evaluation

```bash
# Reproducible eval -- fixed flags, deterministic seed.
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
