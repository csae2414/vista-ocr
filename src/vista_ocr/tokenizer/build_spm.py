"""Train a SentencePiece BPE tokenizer for VISTA-OCR.

The paper says only that they "decreased the vocabulary size" from mBART's
default; concrete number not given. We default to 16k subwords. Special
and spatial tokens are registered as ``user_defined_symbols`` so
SentencePiece never splits them.

Example::

    from vista_ocr.tokenizer.build_spm import train_spm

    train_spm(
        corpus_path="data/processed/corpora/en_pretrain.txt",
        out_prefix="data/processed/vocab/sp_en_16k",
        vocab_size=16000,
        user_symbols=[...],  # vista_ocr.tokenizer.tokenizer.list_special_and_spatial_tokens
    )

The script does not bootstrap a corpus. See ``scripts/bootstrap_tokenizer.py``
for the small generic English corpus we use until PDFA/IDL are downloaded.
"""
from __future__ import annotations

from pathlib import Path

import sentencepiece as spm


def train_spm(
    corpus_path: str | Path,
    out_prefix: str | Path,
    vocab_size: int = 16000,
    user_symbols: list[str] | None = None,
    character_coverage: float = 1.0,
    model_type: str = "bpe",
    max_sentence_length: int = 16384,
) -> Path:
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(corpus_path)
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.Train(
        input=str(corpus_path),
        model_prefix=str(out_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        user_defined_symbols=user_symbols or [],
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        max_sentence_length=max_sentence_length,
        train_extremely_large_corpus=False,
        shuffle_input_sentence=True,
        input_sentence_size=2_000_000,
    )
    model_path = out_prefix.with_suffix(".model")
    if not model_path.exists():
        raise RuntimeError(f"SentencePiece training did not produce {model_path}")
    return model_path
