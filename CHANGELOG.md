# Changelog

User-visible changes per dated entry. Code-internal refactors that
don't affect operators or downstream evaluations are out of scope and
live in commit messages.

## 2026-05-04 — Phase I: stage 1b (unfrozen OCR-only calibration)

Closes reviewer finding #2 (calibration curriculum collapsed).
Paper §3.6.1 splits stage 1 into two sub-stages:

* Stage 1a (frozen-decoder text-only) -- our existing `stage1_run.py`.
* **Stage 1b (this commit)**: unfrozen-all text-only. Decoder warms
  up to the encoder's features without yet being asked to handle
  layout. Bridges the curriculum gap between stage 1 (frozen) and
  stage 2 (unfrozen + multimodal).

Per the second reviewer's note, this stage is NOT gated on Run D's
results -- it's a curriculum-correctness fix, not a speculative
optimization. Lands soon after Phase H regardless of measured ROI
from the data-mix change.

### Added

- `scripts/stage1b_run.py` -- mirrors stage 1 structure with three
  differences: `freeze_decoder=False`, `lambda_text=1.0` (still
  text-only; layout is stage 2's job), `--lr` default 5e-5
  (matching stage 2 -- aggressive 3e-4 would damage the warm
  start). `--init-from` is REQUIRED.
- `vista_ocr.entrypoints.stage1b` -- the CLI mirror.
  ``vista-ocr stage 1b --help`` resolves through the dispatcher.
- Chain envs: `STAGE1B_STEPS` (default 0 -- skip; legacy chain
  unchanged. Set to 10000 for paper-faithful curriculum).
  `STAGE1B_SELECT_ON` (default val_loss; word_f1 climbs slowly when
  the decoder just unfreezes). `STAGE1B_PATIENCE` (default 15).
- Stage 2's `--init-from` now points at stage 1b's `ckpt_final.pt`
  when stage 1b is enabled; falls back to stage 1's when not.
  WARN logged if either ckpt_final.pt is missing.
- Auto-cap extends to stage 1b: when EARLY_STOP=1 and STAGE1B_STEPS
  exceeds the early-stop cap (val_every=500 * (warmup + patience)),
  the chain caps the value with an AUTO-CAP: log line.

### Tests

5 new (3 stage equivalence for stage1b -- `--help` flag-set,
TrainConfig snapshot, model state_dict() keys; 2 chain DRY_RUN --
default skips stage 1b, STAGE1B_STEPS=10000 enables it). All four
stages now share the same equivalence-test contract; the helper
load_checkpoint mock keeps the test offline (stage 1b's required
init-from points at a zero-byte fixture file).

Full suite green (455 passed) + sphinx -W clean.

### Operational note

Run C is still mid-stage-2 on the L40S as of this commit; do NOT
pull on the L40S until Run C completes. Phase H + I land for Run E
when stage 1b is enabled via `STAGE1B_STEPS=10000`.

## 2026-05-04 — Phase H: paper-comparable IDL mix + paper page preset

Closes two of the six paper-divergence findings flagged in the
2026-05-04 code review:

* **#1 Training data mix** (was PDFA-only) -- stages 2 + 3 now
  optionally run on a paper-comparable PDFA + IDL weighted mixture.
  Stage 1 stays PDFA-only by design (frozen-decoder calibration;
  mix adds noise that doesn't help). The mixed loader
  (`make_mixed_pdfa_idl_loader`) already existed with 15 tests; this
  phase just plumbs the flag.
* **#3 Page-resolution preset** (paper appendix says ~2200×1700;
  we capped at 1400×1050) -- adds a `paper` preset behind a 78 GB
  VRAM threshold. Operators with A100-80GB / H100 can opt in via
  `PAGE_PRESET=paper` or `auto`. The lower presets remain because
  our operational hardware tops out at 46 GB; this is an operational
  compromise, not a design choice.

### Added

- `pretrain_chain.sh` envs:
  - `DATA_MIX` (default `pdfa`; `pdfa+idl` enables the mix)
  - `IDL_SHARDS_GLOB` (default `data/raw/idl/idl-train-*.tar`)
  - `IDL_WEIGHT` (default 0.6 -- paper §3.5 leans toward real)
- `--idl-shards` + `--idl-weight` flags on `stage{2,3}_run.py` and
  `vista_ocr.entrypoints.stage{2,3}` (flag-set parity preserved per
  the equivalence-test contract).
- `vista_ocr.training.resolution.PRESETS["paper"]` =
  `PageResolution(2200, 1700)` with a `78 GB` threshold in the
  auto-detect table.
- `paper` added to `--page-preset` choices on every stage / finetune
  / SROIE entrypoint.

### Tests

8 new (6 chain DRY_RUN + 2 resolution): default DATA_MIX=pdfa keeps
no `--idl-shards`; DATA_MIX=pdfa+idl propagates to stages 2+3 only
(stage 1 stays PDFA-only by design); unknown DATA_MIX values reject
loudly; `--page-preset paper` flows through; `IDL_WEIGHT` override
works; missing IDL shards fail fast with a clear error. Resolution
tests pin the 78 GB → paper threshold and the (2200, 1700)
dimensions.

Full suite green (450 passed) + sphinx -W clean.

### Out of scope

- IDL data download (operator's responsibility per box; chain
  expects shards already on disk).
- Mix ratio ablation (60/40 IDL/PDFA chosen as paper-comparable
  starting point per §3.5; tunable via `IDL_WEIGHT`).

### Operational note

Run C is mid-stage-2 on the L40S as of this commit; do NOT pull
on the L40S until Run C completes. Phase H lands for Run D.

## 2026-05-04 — pretrain chain: per-stage select_on + patience + auto-cap

### Symptom

Run C stage 1 hit `EARLY_STOP: reason=patience_exceeded` at step
**10501 / 20000**. Stage 1 has a frozen decoder, so its decoded
output is structurally EOS-only and `val_word_f1` is permanently 0.
The chain forwarded a single `SELECT_ON=val_word_f1` to all three
stages; stage 1 saw "metric never moves" and interpreted it as
"plateau" after the (warmup_vals + patience) val window expired.

### Diagnosis

`scripts/pretrain_chain.sh` had a single `SELECT_ON` env var
forwarded to every stage. The stages have structurally different
metrics:

* Stage 1 (frozen decoder): only `val_loss` moves; `val_word_f1`
  is permanently 0.
* Stages 2-3 (decoder unfrozen): `val_word_f1` is the
  paper-relatable signal; `val_loss` can be misleading
  (DS-fix Phase 3 reasoning).

Plus three adjacent issues this bug surfaced:

* Stage 1's val_loss is bouncier than stages 2-3's word_f1, so the
  early-stop patience that's appropriate for stages 2-3 (10-15) is
  too aggressive for stage 1's signal.
* Stage 2 inits from stage 1's `ckpt_best.pt` -- which is val-loss-
  best, possibly an early transient minimum, not the fully-
  calibrated encoder. Stage 1's `ckpt_final.pt` is what we want.
* When `EARLY_STOP=1` is on, an operator setting `STAGE1_STEPS=N`
  has no way to know that early-stop will fire well before step N.
  Silent budget truncation surfaces only when the operator looks
  for the DONE log line and sees an unexpected step count.

### Fix

Per-stage configuration in `pretrain_chain.sh`:

| Setting | Stage 1 | Stages 2/3 | Why |
|---|---|---|---|
| `STAGE{N}_SELECT_ON` | val_loss | val_word_f1 | structurally |
| `STAGE{N}_PATIENCE` | 40 | 10 / 15 | val_loss is bouncier |
| Init source for next stage | ckpt_final.pt | ckpt_best.pt | last-step state |

Plus:

* **Auto-cap**: when `EARLY_STOP=1` is on and `STAGE_N_STEPS` exceeds
  the early-stop cap, the chain caps the value explicitly with an
  `AUTO-CAP:` log line, so the running budget is observable rather
  than silently truncated.
* **Fallback**: if `ckpt_final.pt` is missing (legacy chain run
  before save_final=True default), stage 2 falls back to
  `ckpt_best.pt` with a `WARN:` log line.
* **DRY_RUN=1 mode**: prints the resolved arglist per stage and
  exits 0 without launching training. Operators can verify env-var
  wiring before committing to a multi-day run.
* **Legacy `SELECT_ON` env var** still works as a whole-chain
  override (back-compat with pre-fix scripts).

6 tests in `tests/test_pretrain_chain_smoke.py`: 3 static-text
contracts (per-stage `--select-on`, defaults pinned, AUTO-CAP block
present) + 3 runtime DRY_RUN tests (each stage's arglist has the
right flags; legacy SELECT_ON propagates; AUTO-CAP fires when
appropriate).

### Operational note

This fix lands while Run C is mid-stage-2 on the L40S. **Do not
pull the fix on the L40S until Run C completes.** The supervisor
wrapper would otherwise resume on different config if the chain
ever restarted.

## 2026-05-03 — `vista-ocr eval` paper-relatable metric blocks

`vista-ocr eval --manifest` now produces text-detection metrics
(DetEval P/R/F1, Area-F1, AP @ IoU 0.5-0.8) when the manifest carries
``bboxes`` per record, and AP @ CER thresholds when
``--cer-ap-thresholds`` is set -- the eval-side metrics paper §4.1
and §4.2 report. Recognition-only manifests are unaffected (sidecar
key set is byte-for-byte identical to pre-G2).

### Added

- ``--bbox-expand-px N`` flag. Expands PREDICTED boxes by ``N`` px on
  each side before detection scoring; ground truth is never expanded.
  Paper §4.1.1 reports SROIE detection numbers with +1/+2 px
  expansion (the model produces tighter boxes than GT). Default 0.
  Asymmetry contract is enforced by 4 tests including the inverse
  case "even at +50 px, F1 cannot exceed 1.0" -- catches a future
  refactor that accidentally expands GT too.
- ``--cer-ap-thresholds`` flag. Comma-separated list of CER
  thresholds; opt-in to the region-OCR AP-at-CER block.
- ``vista_ocr.eval.sidecar.EvalSidecar`` -- single source of truth
  for the JSON sidecar shape. Optional metric blocks
  (``detection``, ``cer_ap``) drop out of the dumped JSON when not
  applicable, so pre-G2 readers keep working.
- ``per_doc_cer`` and ``cer_ap`` in
  ``vista_ocr.eval.metrics_recognition``. ``expand_box`` in
  ``vista_ocr.eval.metrics_detection``.
- Strict manifest homogeneity: the field-set of the first record
  IS the contract; any subsequent record with a different field-set
  is rejected with a clear error. Catches "I accidentally
  interleaved two benchmarks in one manifest."

### Tests

19 new tests on the metric primitives (DetEval split / merged /
disjoint at strict and lenient ``tr``; AP@IoU multi-pred and
intermediate-IoU; CER-AP monotonicity; edge cases on empty inputs
and degenerate boxes); 9 new tests on the eval verb's wiring
(back-compat sidecar key set, detection block populated, the four
bbox-expand asymmetry cases, heterogeneity rejection, CER-AP block
optional). Full suite 431 passed.

## 2026-05-03 — `vista-ocr` CLI + manifest-driven eval/finetune

The package now installs cleanly via pip with a `[project.scripts]`
entry point; operators get a single console script (`vista-ocr`) with
six Python verbs, plus a JSONL manifest schema that lets any benchmark
plug in via a data-prep adapter (no Python plug-in required).

### Added

- **`pyproject.toml` runtime deps + torch extras** (`[cpu]`, `[gpu-cu121]`,
  `[gpu-cu124]`, `[tb]`, `[dev]`). Torch lives in extras only so the
  operator picks the matching wheel via `--extra-index-url`. Install
  recipe in README. Bumped to `0.1.0` (first installable release).
- **`vista-ocr` console script** (`vista_ocr.cli`). Six verbs:
  - ``vista-ocr stage {1|2|3}`` -- pretraining stages, mirrors
    ``scripts/stage{N}_run.py``.
  - ``vista-ocr eval --manifest <jsonl>`` -- generic checkpoint eval.
  - ``vista-ocr finetune --train-manifest --val-manifest --init-from``
    -- generic manifest-driven finetune.
  - ``vista-ocr infer --folder <root>`` -- decode-only; output JSONL
    carries a ``_meta`` header (ckpt path/step, vista-ocr version,
    timestamp) for forensic disambiguation.
  - ``vista-ocr cache`` -- pre-render dataset to a geometry-bound
    cache, mirrors ``scripts/cache_dataset.py``.
- **JSONL manifest schema** (`vista_ocr.data.manifest`). Required:
  ``image`` (relative to manifest dir or absolute), ``ref``. Optional:
  ``bboxes``, ``task``, ``query_text``, ``query_bbox``, ``version``.
  v1; unknown versions are a hard error so silent skips can't produce
  misleading metrics.
- **SROIE manifest emitter** (`scripts/datasets/sroie_to_manifest.py`,
  invoked via ``setup_sroie.sh --emit-manifests <out-dir>``). Walks the
  SROIE flat layout and writes ``train.jsonl`` + ``test.jsonl`` for
  use with the manifest-driven CLI verbs. The legacy path
  (``scripts/benchmarks/sroie/{run.py,eval.py,chain.sh}``) stays in
  place as the proven baseline; the new path runs alongside until both
  produce numerically equivalent SROIE word-F1.

### Behavioural-equivalence guarantees

Drift between the legacy ``scripts/`` and the new
``vista_ocr.entrypoints/`` modules is anchored by three tests per
pretraining stage (``tests/test_stage_equivalence.py``):

1. flag-set parity (``--help`` long-form flags must match).
2. ``TrainConfig`` snapshot equality under a fixed minimal arglist
   with ``train()`` monkeypatched.
3. ``model.state_dict()`` key-set equality (catches architecture
   drift outside ``TrainConfig``).

Plus the cold-start lint (``test_cli_phase_a``,
``test_cli_phase_b``): importing ``vista_ocr.cli`` -- and invoking
``vista-ocr --help`` -- must NOT load torch. Verb modules defer
``import torch`` into ``run()``.

The `cache` entrypoint additionally has a flag-set equivalence test
against ``scripts/cache_dataset.py``.

### Not changed

- `scripts/` is untouched (bit-for-bit identical to before this entry).
  Run A on L40S and Run B on 3090 are mid-flight on those scripts and
  must finish exactly as launched.
- `scripts/benchmarks/sroie/` stays in place. Once the manifest path
  produces numerically equivalent SROIE word-F1 (verification gate,
  deferred), a follow-up commit can retire the SROIE-specific Python
  entry points there in favour of the generic CLI verbs.

## 2026-05-03 — SROIE finetune chain scaffold

Per-benchmark finetune + eval entry points now have their own
home (`scripts/benchmarks/`), parallel to the per-benchmark dataset
adapters in `scripts/datasets/`. The top-level `scripts/` namespace
stays "generic + pretraining" -- no benchmark-specific glue at the
root.

### Added

- **`scripts/benchmarks/sroie/`** -- the SROIE 2019 finetune chain.
  - ``run.py`` loads a pretrained ckpt (typically
    ``checkpoints/stage3/ckpt_best.pt``), trains on the SROIE train
    split with val on the test split, ckpt_best on val_word_f1.
  - ``eval.py`` decodes the full SROIE test split with greedy +
    SROIE-style word-set P/R/F1 (``word_exact_prf``), writes a
    JSON sidecar suitable for pasting into BENCHMARKS.md.
  - ``chain.sh`` orchestrates ``run.py`` then ``eval.py`` in one
    shot; defaults to a paper-style short, low-LR finetune
    (5K steps, lr=1e-5, grad-accum 4, augment on, early-stop on,
    select on val_word_f1).
- **`scripts/benchmarks/README.md`** documents the convention so
  the next benchmark (IAM, FUNSD, ...) drops in cleanly.
- **`vista_ocr.training.val_helpers.sroie_val_batches`** -- mirrors
  ``pdfa_val_batches`` for SROIE; unit-tested in
  ``tests/test_maurdor_sroie.py``.

## 2026-05-03 — Eval methodology, round 2 (DS review blockers)

Three blockers from a data-scientist review. This entry is Phase 1
(val/test split). Phases 2 (decode_n_best second-pass eval) and 3
(select on val_word_f1 not val_loss) follow.

### Changed

- **`ckpt_best` and early-stop now select on `val_word_f1`** by
  default (``CheckpointConfig.select_on``, ``EarlyStopConfig.metric``;
  ``val_loss`` kept as an opt-in via ``--select-on val_loss``). The
  2026-05-02 PDFA hold-out diagnosis showed val_loss can be misleading
  when image conditioning is weak -- the language-model prior dominates
  so val_loss reflects the prior, not OCR quality. word-F1 is what
  BENCHMARKS rows compare against and what the paper reports.

  Mechanically: a ``metric_is_better`` helper centralises direction
  (lower-is-better for val_loss, higher-is-better for word_f1).
  ckpt_best uses a *gate-then-truth* protocol: the cheap n=5 signal
  gates whether to fire the second pass; the n=val_decode_n_best
  result is the *truth* that ratchets ``best_metric`` and writes
  ckpt_best. A noisy gate spike whose truth-pass disagrees logs a
  ``BEST_CANDIDATE_REJECTED:`` line and skips the save -- without
  this filter a noise spike would lock out future real improvements.
  Stage scripts gain ``--select-on``; ``pretrain_chain.sh`` forwards
  ``SELECT_ON``. Startup-time validation hard-fails if
  ``select_on=val_word_f1`` without a configured ``val_decode_fn`` /
  ``val_decode_n > 0`` (otherwise ckpt_best would silently never get
  written). 6 tests in ``tests/test_training.py``.

  Resume back-compat: legacy checkpoints without ``best_metric`` /
  ``best_metric_name`` keys load cleanly; ``best_val_loss`` still
  written at the top level for any external reader.

- **`ckpt_best` candidates get a second-pass low-noise eval**
  (`TrainConfig.val_decode_n_best`, default 256 in stage scripts; 0
  preserves the prior behaviour). Every val pass keeps `decode_n=5`
  so the per-step log line is cheap; when a val pass clears
  ``val_loss < best_val_loss`` the train loop fires a *second*
  ``run_validation`` call with ``decode_n=val_decode_n_best`` samples
  and writes the resulting CER / WER / word-F1 into the checkpoint
  under ``extra["best_candidate"]``. A structured ``BEST_CANDIDATE:``
  log line lands in the supervisor log alongside the existing
  ``EARLY_STOP:`` line so the side-process tailer parses both. Stage
  scripts get a ``--decode-n-best`` flag; ``pretrain_chain.sh``
  forwards ``DECODE_N_BEST``. 2 tests in ``tests/test_training.py``.

  Closes the "ckpt_best is selected against a 5-sample noise floor"
  half of DS-review item #2; the other half (flip the criterion to
  word-F1) is Phase 3.

- **PDFA shards are split into a strict val + a strict test set**
  (`src/vista_ocr/data/split.py`). Shard 0118 is the val shard used
  for ``ckpt_best`` selection during training; shard 0119 is the
  test shard, touched only by ``scripts/eval_run.sh``. Previously
  ``pretrain_chain.sh`` used "the highest-indexed shard" as val,
  meaning the same shard was selected against across all three
  stages -- ckpt_best had effectively been tuned on the held-out
  set after 30+ improvement events. ``stage{1,2,3}_run.py`` call
  ``assert_not_test_shard`` on every shard argument and refuse to
  start when the test shard appears in ``--val-shard`` or
  ``--train-shards``. ``eval_run.sh`` accepts ``TEST_SHARD`` (with a
  one-cycle deprecated ``VAL_SHARD`` shim that warns). 5 tests in
  ``tests/test_split_isolation.py``.

  BENCHMARKS rows generated before this date were against shard
  0119 *as the training-time val shard*, so they are not strict
  held-out test numbers. A header note in ``BENCHMARKS.md`` flags
  this; rows generated after are strict.

## 2026-05-03 — Defect fixes after Run B engineering pass

Three defects from a senior-dev review of the Run B pass.

### Tests

- **Early-stop train_loop integration coverage** (`test_training.py`).
  The 11 unit tests in `test_early_stop.py` covered
  ``early_stop_decision`` in isolation; nothing exercised the wiring
  through ``train()``. Three new integration tests close the gap: a
  flat val curve aborts the loop and emits the structured
  ``EARLY_STOP:`` log line; ``ckpt_final.pt`` carries
  ``reason=early_stop`` plus a round-trippable ``early_stop_state``;
  ``resume_from`` restores the prior counters so a kill+restart does
  not pay another full ``patience * val_every`` window before
  aborting (F2 regression coverage).

### Fixed

- **`Line` re-exported from `vista_ocr.data.types`**. The canonical
  definition stays in `vista_ocr.tokenizer.tokenizer` (avoids a
  circular import with the spatial-token machinery), but data
  modules now import it from `vista_ocr.data.types` so the
  layering smell of ``cache.py`` reaching into the tokenizer
  package is gone. Other data adapters can migrate at their leisure
  -- the old import path remains valid.

- **Mixed loader silently dropped IDL when `len(idl_shards) < num_workers`**
  (`make_mixed_pdfa_idl_loader`). With ``num_workers=8`` and a 2-shard
  IDL set, the previous slice ``shards[worker_id::num_workers]`` left
  workers 2..7 with empty shard lists; the multinomial mixer kept its
  weights but every IDL pull returned ``StopIteration``, so the
  effective mix drifted to PDFA-only as the worker count grew.
  ``_IterableMixedDataset`` now takes ``on_short_shards`` (``"broadcast"``
  default, ``"cap"`` opt-in). Broadcast lets every worker read the full
  short list, and a prime-hashed per-worker cycle seed
  (``(base*100003) ^ (worker_id*31)``) keeps shuffle/cycle streams
  independent. Worker 0 emits a single WARNING when broadcast triggers
  so the operator sees the duplication. 15 new tests in
  ``tests/test_dataloader.py`` (3 seed-helper, 6 slice-policy, 5
  short-shard behaviour, 1 real ``DataLoader`` spawn with
  ``num_workers=2``).

## 2026-05-03 — Run B engineering pass

Eight phases shipped to support a longer + faster Run B on the L40S
without changing the algorithm. Each phase is one commit; tests in
parentheses live in the named files.

### Added

- **Shard cycling** (`PdfaConfig.cycle`, `IdlConfig.cycle`): WebDataset
  ``shardshuffle=True`` + per-sample ``.shuffle(N)`` + ``.repeat()``.
  Stops the "training stalls at 18K steps because data ran out" mode.
  Tests in `tests/test_pdfa.py`.
- **Mixed PDFA + IDL loader** (`make_mixed_pdfa_idl_loader`): weighted
  mixture (default 70/30) over the existing `MixedStream`. Tests in
  `tests/test_dataloader.py`.
- **Bbox-scaling fix in `collate.py`**: bboxes are now scaled by the
  resize factor when `resize_to_canvas` shrinks the image, fixing the
  Albumentations overflow error that fired the first time `--augment`
  was used at `large` preset. Test in `tests/test_data.py`.
- **`--prefetch-factor` flag** on stage scripts; chain forwards
  `NUM_WORKERS`, `PREFETCH_FACTOR`.
- **Production-ready early stopping** (`EarlyStopConfig` with EMA
  smoothing + spike detection + persistent state across resume +
  per-stage `--early-stop-*` flags + structured `EARLY_STOP:` log
  line). 11 tests in `tests/test_early_stop.py`.
- **On-disk sample cache** (`vista_ocr.data.cache`): pre-resized PNG +
  JSON sidecar per sample, geometry-bound manifest, atomic write,
  resume on partial render. Hard-error on geometry mismatch.
  `scripts/cache_dataset.py` populates from PDFA / IDL. 12 tests in
  `tests/test_cache.py`.
- **`--compile` flag** (off by default) on stage scripts; chain
  forwards `COMPILE`. Falls back to eager with a WARNING when
  ``torch.compile`` raises so a multi-day run is never killed by a
  compile glitch. Test in `tests/test_training.py`.
- **`scripts/smoke_chain.sh`**: 600-step (200/200/200) chain smoke
  exercising both stage transitions; runs in ~3-5 min on L40S.
- **HF pre-flight in `pretrain_supervised.sh`**: warms the
  ``naver-clova-ix/donut-base`` download once before any chain
  attempt. Failed network at hour 0 no longer trips a 20-attempt
  restart loop with no useful error.

### Dataset adapters

- `scripts/datasets/` -- per-benchmark prep folder. Each adapter
  takes the dataset's "as distributed" layout and emits the flat
  layout the corresponding loader expects. Convention + planned
  list in `scripts/datasets/README.md`.
- `scripts/datasets/setup_sroie.sh` -- flattens the Kaggle "SROIE
  datasetv2" layout (`train/{img,box,entities}`) to the loader's
  `train/<id>.{jpg,txt}` flat form. Symlinks only -- no file
  copies. Re-runnable.

### Run B launch envs (the recommended set on the L40S)

```
PAGE_PRESET=large GRAD_ACCUM_STEPS=16 INIT_DECODER_FROM=donut
SDPA=1 GRAD_CKPT=0 NUM_WORKERS=8 PREFETCH_FACTOR=8
EARLY_STOP=1
STAGE1_STEPS=50000 STAGE2_STEPS=200000 STAGE3_STEPS=100000
```

Same with ``COMPILE=1`` is the candidate Run C envelope (after Run B
validates the structural changes).

## 2026-05-02 — speed knobs + auto-launcher

After moving the run to an L40S box, two speed knobs were exposed for
the chain so the per-step wall-clock can be cut without touching the
math:

### Added

- `--no-grad-ckpt` flag on `stage{1,2,3}_run.py` (default off, matching
  the prior 3090 behaviour). Disables encoder gradient checkpointing
  for ~30-50 % faster encoder forward at the cost of activation memory.
  Safe on 48 GB+ cards.
- `pretrain_chain.sh` env vars: `SDPA=1` enables the optional C3
  monkey-patch (ship-gate + manifest already in place); `GRAD_CKPT=0`
  forwards `--no-grad-ckpt` to all stage scripts.
- `scripts/launch_when_ready.sh` — polling launcher that waits for
  `--num-pdfa-shards` PDFA + `--num-idl-shards` IDL + the SPM
  tokenizer before firing `pretrain_supervised.sh` in screen.

### Fixed

- `pretrain_supervised.sh` and `pretrain_chain.sh` conda activation:
  they used to look only in `~/miniconda3` (per-user install). Now
  they probe `~/miniconda3`, `/opt/miniconda3`, `/opt/anaconda3`, and
  `~/anaconda3` so a shared install works out of the box. Also the
  activation block now wraps `set +u` ... `set -u` so the activate
  hooks (which reference some unset variables) don't trip the
  supervisor's `set -uo pipefail`.

### Measured (L40S)

- SDPA monkey-patch: **10.05x** kernel speedup, **-17 %** peak memory
  vs eager at the bf16 paper attention shape. Ship-gate manifest
  pinned to `<out>/sdpa_manifest.txt`.

## 2026-05-02 — paper-init investigation

After a 7 h end-to-end pretrain run produced CER 0.84 / word-F1 0.16
on the held-out PDFA shard, a literature comparison (TrOCR, Donut,
DANIEL, VISTA-OCR) surfaced three independently impactful settings
the run was missing relative to the paper:

- **Decoder initialisation.** Paper §3.2 initialises the decoder
  from `naver-clova-ix/donut-base`; ours was random.
- **Effective batch size.** Paper uses ~11 on A100-80GB; we used 1.
- **Data mixture.** Paper trains on ~120 K synthetic + ~170 K real;
  we trained on PDFA only.

This release adds the infrastructure to address all three on the next
pretraining run.

### Added

- `vista_ocr.models.donut_init.init_decoder_from_donut` — load the
  BART text-decoder body from `naver-clova-ix/donut-base` into our
  `MBartDecoder`. Vocab-shaped tensors (`embed_tokens`, `lm_head`,
  `final_logits_bias`) are dropped (different vocab); transformer
  body, cross-attention, layer norms, positional embeddings transfer.
  Refuses to silently fall back to random init if fewer than
  `require_min_overlap` tensors copied. 13 unit tests.
- `--init-decoder-from {random,donut}` flag on `stage1_run.py`. The
  default stays `random` for back-compat; `donut` is the paper-faithful
  setting and the new chain default.
- `--grad-accum-steps N` flag on all three stage scripts (default 1
  for back-compat). `pretrain_chain.sh` defaults to 8 via env.
- `vista_ocr.data.mixture_stream.MixedStream` — weighted mixture
  across multiple sample iterators. Composes with the existing
  `TaskMix` (which decides the *task* per sample). 8 unit tests.
- `CheckpointConfig.save_final` (default `True`): writes
  `ckpt_final.pt` at end of training alongside `ckpt_best.pt`.
  Final captures last-step state; best is val-loss-selected (which
  this run showed can be misleading). 2 tests.
- `scripts/eval_run.sh` — reproducible eval wrapper. Fixes shard,
  decode flags, and seed; the only argument that changes is the
  checkpoint. Use this when generating BENCHMARKS.md rows.
- `pretrain_chain.sh` rewritten: tunables via env vars; defaults flip
  to `INIT_DECODER_FROM=donut`, `GRAD_ACCUM_STEPS=8`, `AUGMENT=1`.
  All operator-visible knobs documented in the script header.
- `baselines/random_init_20260502/` — frozen comparison artefacts
  (ckpt + eval json + commit hash) for the previous pretrain run.

### Fixed

- `MBartDecoder.generate_greedy` now passes `encoder_hidden_states`
  through the per-step model forward via a hand-rolled greedy loop;
  HF's `MBartForCausalLM.prepare_inputs_for_generation` strips them
  silently, which made every input image produce identical greedy
  output. Same class of fragility as HF #28005.

### Diagnostics

- `eval_pdfa_holdout.py` gains `--debug-dump`, `--repetition-penalty`,
  `--no-repeat-ngram-size`, `--max-new-tokens`, `--min-new-tokens`.
- BENCHMARKS.md gains a per-condition comparison table for the
  PDFA hold-out and a paragraph documenting the diagnosis.
