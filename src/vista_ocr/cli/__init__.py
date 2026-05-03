"""``vista-ocr`` console-script entry point.

Phase A: minimal stub. Only ``--help`` and ``--version`` are wired up;
later phases register subcommands (eval / finetune / infer / cache /
stage) into this dispatcher.

Cold-start lint contract: importing this module MUST NOT load torch.
The CLI is invoked many times per session for ``--help`` and shouldn't
pay the ~1.5 s torch-import tax unless a verb that needs it actually
runs. Subcommand modules defer ``import torch`` into ``run()``.
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
    parser = argparse.ArgumentParser(
        prog="vista-ocr",
        description=(
            "VISTA-OCR command-line interface. Subcommands are added in "
            "later phases (eval, finetune, infer, cache, stage)."
        ),
    )
    parser.add_argument(
        "--version", action="version",
        version=f"vista-ocr {_resolve_version()}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    # No subcommands yet: print help and return 0 so the entry point
    # is observable as installed.
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
