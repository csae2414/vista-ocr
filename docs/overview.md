# Overview

This project implements VISTA-OCR end-to-end:

- **Encoder** — DANIEL-style FCN (`vista_ocr.models.encoder`).
- **Decoder** — `mbart-large-50` decoder, vocabulary reduced to EN + spatial tokens (`vista_ocr.models.decoder`).
- **Tokenizer** — SentencePiece subwords plus the three serialization schemes from paper Table 7 (`vista_ocr.tokenizer`).
- **Loss** — `λ · L_text + (1−λ) · L_loc` with prompt-token masking (`vista_ocr.training.losses`).
- **Data** — PDFA + IDL via WebDataset; SynthDOG-bbox and SROIE-synth synthetic generators (`vista_ocr.data`).
- **Training** — Accelerate-based loop with bf16 + grad accumulation (`vista_ocr.training.train_loop`).
- **Eval** — CER, WER, word-exact F1, Wolf & Jolion DetEval, AP@IoU, Area-F1 (`vista_ocr.eval`).
- **Inference** — greedy / beam decoding plus output parser (`vista_ocr.inference.generate`).

See `PLAN.md` for the paper-faithfulness audit and the explicit list of
hyperparameters that are *not* stated in the paper and therefore guessed.
