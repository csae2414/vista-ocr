"""Phase A smoke tests for the ``vista-ocr`` console script.

* ``--version`` exits 0 and prints something sensible.
* Importing ``vista_ocr.cli`` does not load torch (cold-start lint).

The torch-import lint is the load-bearing test: every later phase
registers subcommand modules that DO need torch, but they must defer
``import torch`` into ``run()`` so the cli dispatcher itself stays
fast. A regression here would silently make ``vista-ocr --help`` slow
by ~1.5 s.
"""
from __future__ import annotations

import subprocess
import sys


def test_cli_help_exits_zero(capsys):
    """In-process: ``main(['--help'])`` exits cleanly via SystemExit(0)."""
    import pytest as _pytest

    from vista_ocr.cli import main

    with _pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "vista-ocr" in out


def test_cli_version_prints_something(capsys):
    import pytest as _pytest

    from vista_ocr.cli import main

    with _pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    # argparse writes "version" actions to stdout; format is
    # ``vista-ocr <version>\n``.
    assert out.startswith("vista-ocr ")


def test_importing_cli_does_not_load_torch():
    """Cold-start lint: a fresh subprocess that imports vista_ocr.cli
    must NOT have torch in sys.modules afterwards.

    Using a subprocess (instead of asserting in the current process)
    is critical: pytest itself imports torch via fixtures elsewhere
    in the test suite, so an in-process check is a tautology.
    """
    code = (
        "import sys; "
        "import vista_ocr.cli; "
        "assert 'torch' not in sys.modules, "
        "    'vista_ocr.cli must not import torch at module-load time'"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, (
        f"cold-start lint failed: {r.stdout}\n{r.stderr}"
    )
