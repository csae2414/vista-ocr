# Dataset adapters

One script per benchmark dataset. Each takes the dataset's "as
distributed" layout (Kaggle re-upload, FKI Bern registration tarball,
ELRA delivery, ...) and emits the flat layout the corresponding
loader in `vista_ocr/data/` expects.

## Why a separate folder

Top-level `scripts/` is for **dataset-agnostic** tools: chain
runners, eval wrappers, cache builders, profilers. As we add more
benchmarks, the per-dataset prep scripts pile up and would otherwise
swamp those. Keeping them here means:

- Operators looking for a chain runner don't sift through ten
  `setup_<dataset>.sh` files.
- License caveats (some datasets are paywalled, some require
  registration) stay grouped where the operator hits them.
- The chain runners that **use** these adapters (e.g.
  `scripts/finetune_chain.sh`) call them by relative path.

## Currently available

| Script | Source | Loader | License |
|---|---|---|---|
| `setup_sroie.sh` | Kaggle "SROIE datasetv2" (Urban Knupleš) — community re-upload of ICDAR 2019 SROIE | `vista_ocr.data.sroie` | Original ICDAR challenge license |

## Planned

- `setup_iam.sh` — IAM Handwriting DB. Free with FKI Bern registration.
- `setup_maurdor.sh` — MAURDOR. ELRA, paid.
- `setup_pagexml.sh` — generic PageXML / Alto → sample-cache producer.

## Convention

Each adapter:

1. Takes a source root as its first arg.
2. Writes to `data/raw/<dataset>/` by default, optional override as second arg.
3. Uses **symlinks** rather than copies when possible (dataset is
   typically read-only).
4. Verifies counts and prints a `DONE:` line on success.
5. Re-runnable safely (clobbers existing symlinks at the target).

When in doubt, mirror `setup_sroie.sh`.
