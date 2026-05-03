"""``vista-ocr`` CLI verb implementations.

Each module in this package exposes the trio:

* ``build_parser(*, add_help: bool = True) -> argparse.ArgumentParser``
* ``run(args: argparse.Namespace) -> int``
* ``main(argv: list[str] | None = None) -> int``

The ``vista_ocr.cli`` dispatcher wires each verb's ``build_parser``
into a top-level subparser via ``parents=[…]`` (each parser passes
``add_help=False`` when used as a parent). Defer ``import torch`` and
any expensive setup into ``run()`` so the dispatcher stays cold-start
fast (see the cold-start lint in ``tests/test_cli_phase_a.py``).

Several verbs (``cache``) mirror an existing ``scripts/`` script for
backwards compatibility with operators who run the legacy paths;
``tests/test_entrypoints_equivalence.py`` is the contract that
prevents drift.
"""
