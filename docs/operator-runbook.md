# Operator runbook

End-to-end recipe for taking a fresh box, downloading data, training
the chain, and getting a PDFA hold-out number into BENCHMARKS.md.

## 1. Environment

```bash
# Plain pip + matching torch wheel (preferred; conda not required):
pip install -e ".[gpu-cu121]" --extra-index-url https://download.pytorch.org/whl/cu121
vista-ocr --version
pytest -q                   # ~2 minutes, ~390 tests passing
```

The `vista-ocr` console script exposes the Python verbs (stage 1/2/3,
eval, finetune, infer, cache); the legacy `python scripts/...` and
chain wrappers in `scripts/*.sh` keep working unchanged. Operators
mid-run are unaffected.

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

### Per-stage early-stop config

`pretrain_chain.sh` configures `--select-on` and
`--early-stop-patience` per stage; stage 1 differs from stages 2-3
because its frozen decoder makes `val_word_f1` structurally 0:

| Setting | Stage 1 | Stages 2/3 |
|---|---|---|
| `STAGE{N}_SELECT_ON` | `val_loss` | `val_word_f1` |
| `STAGE{N}_PATIENCE` | 40 | 10 / 15 |

Override per-stage with `STAGE1_SELECT_ON=...` etc. Override
whole-chain with `SELECT_ON=...` (back-compat).

When `EARLY_STOP=1`, `STAGE_N_STEPS` is auto-capped if early-stop
would fire before the configured budget; an `AUTO-CAP:` log line
surfaces the change. Run with `DRY_RUN=1` to see the resolved
arglist per stage without launching training.

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

### Phase J: synthetic handwritten data (opt-in)

Phase J adds a license-clean synthetic handwritten line generator
that mixes into stages 2+3 alongside PDFA (and optional IDL). Stage
1 + 1b remain PDFA-only. Synth is opt-in; default `DATA_MIX=pdfa`
behaviour is bit-identical to pre-Phase-J.

**Honest framing:** this is a synthetic approximation for
distribution coverage, NOT a paper-equivalent reproduction of
synthetic IAM/RIMES; per-writer variation is not modeled. Do not
read the IAM row as paper-comparable until the post-J A/Bs land
(see `BENCHMARKS.md`).

#### Stage corpora + fonts (one-time)

The runtime never downloads. Stage corpora locally first (the
acquisition step is the only place HF dataset names appear):

```bash
./scripts/datasets/setup_synth_corpus.sh   # writes corpora/synth/en/*.txt
# Fonts: download 8-12 OFL/Apache handwritten TTFs into
# /path/to/handwritten/fonts/. PROVENANCE.json under
# vista_ocr/data/synth/fonts/handwritten/ tracks which fonts have
# been bundled (license + SHA256 binding); start there.
```

#### Run with synth on

```bash
DATA_MIX=pdfa+synth \
SYNTH_TEXT_CORPUS_EN=corpora/synth/en/pg19.txt \
SYNTH_FONT_DIR=/path/to/handwritten/fonts \
SYNTH_WEIGHT=0.2 \
  ./scripts/pretrain_chain.sh
```

`DATA_MIX` matrix (the four supported shapes):

| `DATA_MIX` | PDFA | IDL | Synth |
|---|---|---|---|
| `pdfa` (default) | 100% | — | — |
| `pdfa+idl` | `1 - IDL_WEIGHT` | `IDL_WEIGHT` (default 0.6) | — |
| `pdfa+synth` | `1 - SYNTH_WEIGHT` | — | `SYNTH_WEIGHT` (default 0.2) |
| `pdfa+idl+synth` | `1 - IDL_WEIGHT - SYNTH_WEIGHT` | `IDL_WEIGHT` | `SYNTH_WEIGHT` |

The chain hard-fails if `IDL_WEIGHT + SYNTH_WEIGHT >= 1.0` (which
would yield `pdfa_frac <= 0`). Reduce one of the two.

#### Tokenizer audit before French (gate for J1b)

Production SPM is English-heavy. Run the FR-coverage audit BEFORE
adding French synth — otherwise the model trains to predict `<unk>`
on French text, not French OCR.

```bash
python tools/audit_fr_coverage.py \
  --spm data/processed/vocab/sp_en_16k.model \
  --corpus corpora/synth/fr/wikitext_fr.txt
```

Exit 0 = `<= 0.5%` `<unk>` AND every FR accent / ligature / curly
apostrophe encodes without `<unk>`. Exit 1 = block J1b until SPM is
retrained with a FR-mixed corpus.

#### Distribution sanity bands

```bash
python tools/synth_target_distributions.py \
  --source pdfa --shards 'data/raw/pdfa/pdfa-eng-train-{0000..0117}.tar' \
  --out notes/synth_target_distributions.json --n 5000
```

PDFA / IDL bands are **caveats** (catch pathological synth output
like avg chars/line=80 vs real=25), NOT fitting targets. IAM bands
are the actual fit target if/when IAM is accessible.

#### OCR-vs-layout ablation

For IAM, recognition can matter more than bbox layout during
pretraining. Flip the synth-side task tag:

```bash
DATA_MIX=pdfa+synth SYNTH_TASK=ocr ...   # synth emits task="ocr"
```

If post-J IAM A/B with `SYNTH_TASK=ocr` beats `SYNTH_TASK=ocr_layout`
by ≥ 2 points, flip the chain default. CHANGELOG records the flip.

#### Stage-3 task histogram shift when synth is on (load-bearing)

Stage 3 normally introduces all four tasks (ocr / ocr_layout /
region_ocr / find_it) at uniform 25/25/25/25 weights via
`MixedTaskStream`. **Synth-handwritten samples are carved out of
that relabelling** (see `vista_ocr.data.mixture` module docstring;
the carve-out exists so `--synth-task` survives end-to-end into
the post-J OCR-vs-layout ablation).

The carve-out shifts the effective stage-3 task histogram when
synth is on:

| Mode | ocr | ocr_layout | region_ocr | find_it |
|---|---|---|---|---|
| Synth off (default) | 0.25 | 0.25 | 0.25 | 0.25 |
| `SYNTH_WEIGHT=0.2`, `SYNTH_TASK=ocr_layout` | 0.20 | **0.40** | 0.20 | 0.20 |
| `SYNTH_WEIGHT=0.2`, `SYNTH_TASK=ocr`        | **0.40** | 0.20 | 0.20 | 0.20 |

**Implication for the post-J A/B protocol:** turning synth on AND
turning synth off changes BOTH (a) the data content and (b) the
task histogram. A naive synth-on-vs-off comparison conflates the
two effects.

If you want a clean "synth content presence" comparison
independent of task-histogram shape, pass non-default `--w-*`
weights so both arms produce the same effective histogram. For
example, with `SYNTH_TASK=ocr_layout, SYNTH_WEIGHT=0.2`, run
stage 3 with `--w-ocr-layout 0.0625 --w-ocr 0.3125 --w-region-ocr 0.3125
--w-find-it 0.3125` so the on-synth arm's effective histogram lands
back at 25/25/25/25.

Document which protocol you ran in the BENCHMARKS row notes; the
two protocols answer different questions (joint effect vs synth-
content-only).

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

For benchmark rows that use the manifest-driven path
(`vista-ocr eval --manifest`), the JSON additionally carries a
``detection`` block (DetEval P/R/F1, Area-F1, AP @ IoU) when the
manifest has bboxes. Use ``--bbox-expand-px 2`` to reproduce paper
§4.1.1's SROIE detection numbers (the model produces tighter boxes
than GT; predicted boxes are expanded, GT is not). For
region-OCR rows, opt in to AP-at-CER via
``--cer-ap-thresholds 0.0,0.1,0.2,0.3``.

### What `ckpt_best.pt` means

Since 2026-05-03, ``ckpt_best.pt`` is selected on **val_word_f1**
(higher-is-better) measured on the second-pass eval at
``--decode-n-best`` samples (default 256). The legacy val_loss
selector is opt-in via ``--select-on val_loss``. The selection
protocol is:

1. Every val pass scores ``decode_n=5`` samples (cheap; for the
   per-step log line).
2. When the cheap word-F1 *appears* to improve over ``best_metric``,
   a second eval pass scores ``--decode-n-best`` samples on the same
   val shard.
3. ckpt_best is written *only* if the second-pass word-F1 still
   beats ``best_metric``. A noisy cheap-signal spike that the
   second pass disagrees with logs ``BEST_CANDIDATE_REJECTED:`` and
   skips the save (so a future real improvement is not locked out
   by the noise).

The structured ``BEST_CANDIDATE:`` line written on every accepted
save carries the persisted (n=256) CER / WER / word-F1 numbers --
those are what BENCHMARKS rows are read off, not the per-val n=5.

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
