"""J0 #2: tokenizer French-coverage audit.

Run on the box that owns the production SPM model (currently the L40S
at /opt/vista-ocr/). Reports `<unk>` rate over a French sample and
per-character coverage of the full set of accents, ligatures, and
curly apostrophe that French handwriting will emit. Gates J1b
(notes/plan_phase_j.md §0/J0).

Usage:
    python tools/audit_fr_coverage.py --spm <path/to/spm.model> \\
        [--corpus <path/to/fr_text.txt>] [--n 1000]

If --corpus is omitted, a small in-tree fixture is used; the absolute
<unk> rate from a fixture is less reliable than from a real FR corpus,
so prefer --corpus when possible.

Exit code is 0 if coverage clears the J1b gate (≤ 0.5% <unk>, all
target chars present), 1 otherwise. Writes a JSON report to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# These match HandwrittenLineSynth.FRENCH_CHARSET; duplicated here so
# the audit tool has zero deps on the package being importable.
FR_CHARS = set("éèêëàâäçôöïîùüÿæœÉÈÊËÀÂÄÇÔÖÏÎÙÜŸÆŒ’")

FIXTURE = """\
Œuvres choisies du grand maître. L’ancienne église était fermée.
Voilà l’été qui s’en va déjà ! Il faisait très chaud, très fâché.
Cœur, sœur, mœurs, fœtus, bœuf — voici des mots français usuels.
Çà et là, des œufs frais. Un naïf entêté, ô surprise du destin !
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spm", required=True, type=Path,
                   help="Path to the production SPM model (.model file).")
    p.add_argument("--corpus", type=Path, default=None,
                   help="Path to a UTF-8 French corpus, one sentence per line.")
    p.add_argument("--n", type=int, default=1000,
                   help="Number of sentences to tokenise (default 1000).")
    p.add_argument("--unk-threshold", type=float, default=0.005,
                   help="Maximum <unk> rate to clear the gate (default 0.005 = 0.5%%).")
    return p.parse_args()


def load_text(path: Path | None, n: int) -> list[str]:
    if path is None:
        return [s.strip() for s in FIXTURE.splitlines() if s.strip()]
    raw = [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    return raw[:n]


def main() -> int:
    args = parse_args()

    try:
        import sentencepiece as spm
    except ImportError:
        print(json.dumps({"error": "sentencepiece not installed; run on the SPM box"}), file=sys.stderr)
        return 1

    if not args.spm.exists():
        print(json.dumps({"error": f"SPM model not found: {args.spm}"}), file=sys.stderr)
        return 1

    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.spm))
    unk_id = sp.piece_to_id("<unk>")

    sentences = load_text(args.corpus, args.n)
    if not sentences:
        print(json.dumps({"error": "empty corpus"}), file=sys.stderr)
        return 1

    total_pieces = 0
    unk_pieces = 0
    for s in sentences:
        ids = sp.EncodeAsIds(s)
        total_pieces += len(ids)
        unk_pieces += sum(1 for i in ids if i == unk_id)

    char_covered: dict[str, bool] = {}
    for c in sorted(FR_CHARS):
        ids = sp.EncodeAsIds(c)
        # A char is "covered" if no token in its encoding is <unk>.
        char_covered[c] = all(i != unk_id for i in ids)

    unk_rate = unk_pieces / total_pieces if total_pieces else 1.0
    missing_chars = sorted(c for c, ok in char_covered.items() if not ok)
    gate_passed = unk_rate <= args.unk_threshold and not missing_chars

    report = {
        "spm_model": str(args.spm),
        "corpus": str(args.corpus) if args.corpus else "(in-tree fixture)",
        "n_sentences": len(sentences),
        "total_pieces": total_pieces,
        "unk_pieces": unk_pieces,
        "unk_rate": unk_rate,
        "unk_threshold": args.unk_threshold,
        "fr_chars_total": len(FR_CHARS),
        "fr_chars_covered": sum(char_covered.values()),
        "fr_chars_missing": missing_chars,
        "gate_passed": gate_passed,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
