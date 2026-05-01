# Overview

This project implements VISTA-OCR end-to-end:

- **Encoder** — DANIEL-style FCN (`vista_ocr.models.encoder`).
- **Decoder** — `mbart-large-50` decoder, vocabulary reduced to EN + spatial tokens (`vista_ocr.models.decoder`).
- **Tokenizer** — SentencePiece subwords plus the three serialization schemes from paper Table 7 (`vista_ocr.tokenizer`).
- **Loss** — `λ · L_text + (1−λ) · L_loc` with prompt-token masking (`vista_ocr.training.losses`).
- **Data** — PDFA + IDL via WebDataset; SynthDOG-bbox and SROIE-synth synthetic generators (`vista_ocr.data`); a frozen `BBox` value object centralises bbox geometry; `PdfaShardReader` deduplicates JSON-sidecar walking across helper scripts.
- **Training** — bf16 autocast + grad checkpointing + KV-cache decoding (`vista_ocr.training.train_loop`); VRAM-aware page-resolution presets via `--page-preset` (`vista_ocr.training.resolution`).
- **Optional SDPA** — opt-in `MBartAttention` monkey-patch (`vista_ocr.models.sdpa_patch`) with a CI-runnable ship-gate covering forward, backward, KV-cache, beam log-prob, and bf16 at the paper attention shape.
- **Ablation** — shared `Ablation` base class (`vista_ocr.ablation`) so adding a new comparison script is a subclass plus an argparse shim.
- **Eval** — CER, WER, word-exact F1, Wolf & Jolion DetEval, AP@IoU, Area-F1 (`vista_ocr.eval`).
- **Inference** — greedy / beam decoding plus output parser (`vista_ocr.inference.generate`).

See `README.md` for the paper-faithfulness audit and the explicit list of
hyperparameters that are *not* stated in the paper and therefore guessed.
