# Synth font + text-source holdout

`HandwrittenLineSynth` ships an explicit holdout split so a synth-only
validation manifest can detect overfit to font identity / templated
language. **What this measures:** synthetic-overfit detection. **What
it does NOT measure:** real-handwriting transfer — that is the IAM
A/B in `notes/plan_phase_j.md` Post-J kill criterion.

## Held-out fonts

(populate once fonts are bundled — pick 2 out of 10 that are visually
distinct from the rest)

| Font | Reason |
|---|---|
| _TBD_ | _decorative-but-allowed; out of training distribution_ |
| _TBD_ | _strongly cursive; reserved for synth-val_ |

## Held-out text sources

(populate once corpora are staged via
`scripts/datasets/setup_synth_corpus.sh`)

| Source tag | Path | Reason |
|---|---|---|
| _TBD_ | _corpora/synth/en/holdout.txt_ | _disjoint sentences, never used in train_ |

## Test enforcement

- `tests/test_synth_handwritten.py` asserts the holdout fonts +
  text-source tag are NEVER drawn by a `HandwrittenLineSynth`
  configured with `holdout=False` (default).
- The synth-val manifest builder uses `holdout=True` and points at
  ONLY the held-out fonts and text-source.
