# Changelog

User-visible changes per dated entry. Code-internal refactors that
don't affect operators or downstream evaluations are out of scope and
live in commit messages.

## 2026-05-03 — Defect fixes after Run B engineering pass

Three defects from a senior-dev review of the Run B pass.

### Fixed

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
