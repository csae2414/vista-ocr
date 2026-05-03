"""``vista-ocr`` console-script entry point.

Subcommands compose by importing each verb's ``build_parser(add_help=False)``
into a top-level subparser via argparse's ``parents=[…]`` mechanism.
The verb modules themselves live under :mod:`vista_ocr.entrypoints`.

Cold-start lint contract (tests/test_cli_phase_a.py): importing this
module MUST NOT load torch. Verb modules defer ``import torch`` into
their ``run()`` so ``vista-ocr --help`` stays sub-second.
"""
from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version


def _resolve_version() -> str:
    try:
        return version("vista-ocr")
    except PackageNotFoundError:
        return "unknown (not installed)"


def build_parser() -> argparse.ArgumentParser:
    """Top-level CLI parser. Each subverb is wired in lazily by
    importing its ``build_parser`` and using ``parents=[…]``.

    Verb module imports are intentionally inside this function (not
    at module top-level) so that ``import vista_ocr.cli`` stays cheap
    -- the cold-start lint asserts torch isn't loaded by an import
    of this module. We pay the verb-module import cost only when the
    operator actually invokes the CLI (and even then, the verbs
    themselves still defer ``import torch`` into ``run()``).
    """
    from vista_ocr.entrypoints import (
        cache as _cache,
        eval_manifest as _eval,
        finetune_manifest as _finetune,
        infer_folder as _infer,
    )

    parser = argparse.ArgumentParser(
        prog="vista-ocr",
        description="VISTA-OCR command-line interface.",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"vista-ocr {_resolve_version()}",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    _attach(sub, "eval", _eval, "Evaluate a checkpoint against a JSONL manifest.")
    _attach(sub, "finetune", _finetune, "Finetune from a pretrained ckpt + manifest pair.")
    _attach(sub, "infer", _infer, "Decode every image in a folder (no GT).")
    _attach(sub, "cache", _cache, "Pre-render dataset samples to a geometry-bound cache.")

    return parser


def _attach(sub, name: str, mod, help_text: str) -> None:
    """Compose the verb's parser as a subparser. The verb module owns
    its flag definitions; we only forward and remember which run() to
    call when the user picks this verb."""
    parent = mod.build_parser(add_help=False)
    sp = sub.add_parser(name, parents=[parent], help=help_text,
                        description=help_text)
    sp.set_defaults(_verb_run=mod.run)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    verb_run = getattr(args, "_verb_run", None)
    if verb_run is None:
        parser.print_help()
        return 0
    return verb_run(args)


if __name__ == "__main__":
    sys.exit(main())
