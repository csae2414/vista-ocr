# Changelog

User-visible changes per dated entry. Code-internal refactors that
don't affect operators or downstream evaluations are out of scope and
live in commit messages.

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
