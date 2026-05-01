# Contributing

Thanks for thinking about contributing. The repository tracks a paper
reimplementation, so please pair a code change with the relevant section
of the paper or with empirical evidence on the GPU.

## Setup

```bash
conda env create -f environment.yml
conda activate vista-ocr
pip install -e .
pytest -q
```

You should see **101 tests passing** before you change anything.

## What kinds of PRs are welcome

- **Faithfulness improvements.** If a paper detail is incorrectly
  modelled, point at the paper section and propose the fix.
- **Speed wins.** PRs that make `bench_dataloader.py` faster are
  welcome. Document the win in the commit message with reproducible
  numbers.
- **Dataset coverage.** Loaders for new public OCR datasets in the same
  shape as `vista_ocr.data.iam` / `sroie` / `maurdor`.
- **Bugfixes** with regression tests.
- **Reproducibility numbers.** If you run pretraining + finetune to
  completion on a real GPU, please open a PR adding your numbers to a
  `BENCHMARKS.md` table.

## What kinds of PRs are unlikely to merge

- Renames or stylistic refactors with no behavioural change.
- New optional dependencies without a clear win documented in the
  benchmark script.
- Changes to defaults in `configs/base.yaml` without empirical evidence
  on a real GPU run.

## Conventions

- **Tests:** every new module gets a `tests/test_<name>.py` file.
- **Logging:** every module starts with `LOG = logging.getLogger(__name__)`.
- **Docstrings:** Sphinx (Google or NumPy style) — these go straight
  into the API docs via autodoc.
- **Type hints:** required on public function signatures.
- **Linting:** `ruff check src tests` must pass.
- **Comments:** explain *why*, not *what*. Reference the paper section
  or PR/issue number where useful.
- **Commit messages:** "subject line + paragraph explaining why" — see
  the existing history for the style.

## License

By contributing, you agree your contributions are licensed under the MIT
licence (see `LICENSE`).
