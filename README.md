# vista-ocr

A from-scratch reimplementation of **VISTA-OCR** (Hamdi, Tamasna, Boisson, Paquet — *VISTA-OCR: Towards generative and interactive end to end OCR models*, arXiv:[2504.03621](https://arxiv.org/abs/2504.03621), Apr 2025). No official code has been released; this repo follows the paper as faithfully as possible.

The paper specifies architecture and curriculum but omits most hyperparameters. See `PLAN.md` for the full implementation plan, the documented gaps, and our defaults.

## Status

Pre-implementation: scaffold only. Tokenizer, model, training and eval are not yet built. Track progress against the task list in `PLAN.md` §3.

## Setup

```bash
# CPU dev box (this machine)
conda env create -f environment.yml
conda activate vista-ocr
pip install -e .

# GPU VM (later)
conda env create -f environment-cuda.yml
```

## Layout

See `PLAN.md` §3 for the directory tree and per-module responsibilities.

## License

MIT — see `LICENSE`.
