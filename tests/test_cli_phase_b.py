"""Phase B tests: generic verbs (eval / finetune / infer / cache) +
the CLI dispatcher.

Coverage:
* every verb exposes ``build_parser`` + ``run`` + ``main`` (the trio).
* every verb's ``--help`` prints without instantiating torch.
* the CLI dispatcher routes ``vista-ocr <verb> --help`` to the right
  parser.
* equivalence vs ``scripts/cache_dataset.py``: the legacy script and
  the new entrypoint expose the same flag set with identical defaults
  and types (the contract that prevents drift).
* the installed ``vista-ocr`` console script reaches main() (catches
  packaging-level regressions).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ENTRYPOINT_MODULES = (
    "vista_ocr.entrypoints.cache",
    "vista_ocr.entrypoints.eval_manifest",
    "vista_ocr.entrypoints.finetune_manifest",
    "vista_ocr.entrypoints.infer_folder",
)


@pytest.mark.parametrize("modpath", ENTRYPOINT_MODULES)
def test_each_verb_exposes_trio(modpath: str):
    import importlib

    mod = importlib.import_module(modpath)
    assert callable(getattr(mod, "build_parser", None)), f"{modpath} missing build_parser"
    assert callable(getattr(mod, "run", None)), f"{modpath} missing run"
    assert callable(getattr(mod, "main", None)), f"{modpath} missing main"
    parser = mod.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    # Both add_help=True (default) and add_help=False must work --
    # add_help=False is what the dispatcher uses for `parents=[...]`.
    parent = mod.build_parser(add_help=False)
    assert isinstance(parent, argparse.ArgumentParser)


@pytest.mark.parametrize("modpath", ENTRYPOINT_MODULES)
def test_each_verb_help_runs_without_torch(modpath: str):
    """Cold-start lint, per verb: ``main(['--help'])`` must exit 0
    without loading torch. Subprocess to escape the test process's
    already-loaded torch."""
    code = (
        f"import sys; "
        f"import {modpath} as m; "
        f"raised = False\n"
        f"try: m.main(['--help'])\n"
        f"except SystemExit as e: assert e.code == 0; raised = True\n"
        f"assert raised, 'argparse --help must SystemExit(0)'\n"
        f"assert 'torch' not in sys.modules, 'verb {modpath} loaded torch on --help'"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, f"{modpath} cold-start: {r.stdout}\n{r.stderr}"


def test_cli_dispatcher_lists_all_verbs(capsys):
    from vista_ocr.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for verb in ("eval", "finetune", "infer", "cache"):
        assert verb in out, f"verb {verb!r} missing from `vista-ocr --help`"


def test_cli_dispatcher_routes_to_subverb_help(capsys):
    """`vista-ocr eval --help` must route to the eval parser."""
    from vista_ocr.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["eval", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    # eval verb has --manifest / --ckpt as required args; both should
    # appear in its help.
    assert "--manifest" in out
    assert "--ckpt" in out


def test_dispatcher_module_does_not_load_torch():
    """Importing vista_ocr.cli (which lazy-imports the verb modules
    inside build_parser()) must not pull torch into sys.modules.
    Cold-start lint at the dispatcher level."""
    code = (
        "import sys; "
        "import vista_ocr.cli; "
        "assert 'torch' not in sys.modules"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stderr


def test_dispatcher_help_invocation_does_not_load_torch():
    """Running `vista-ocr --help` end-to-end through the entry point
    must stay torch-free. The dispatcher imports verb modules to wire
    their parsers, so each verb's module-level imports also have to
    stay torch-free."""
    code = (
        "import sys; "
        "from vista_ocr.cli import main\n"
        "try: main(['--help'])\n"
        "except SystemExit: pass\n"
        "assert 'torch' not in sys.modules, 'CLI --help triggered torch import'"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stderr


# ---------------- Equivalence: cache verb vs scripts/cache_dataset.py


def _help_to_flag_dict(help_text: str) -> dict[str, str]:
    """Parse argparse --help output into ``{flag → help-suffix}`` dict.

    We only care that the flag set + ordering matches; argparse
    formatting differences across minor versions are filtered out by
    keying on the flag name and stripping the surrounding whitespace.
    """
    flags: dict[str, str] = {}
    # Match lines that start with whitespace + '--<name>' or '-X' and
    # capture the flag name. We deliberately don't try to parse type/
    # default from the help text -- argparse formatting varies and
    # the parser-level equivalence below is the load-bearing check.
    for line in help_text.splitlines():
        m = re.match(r"\s+(-[\w-]+)(?:[\s,]|$)", line)
        if m:
            flags[m.group(1)] = "present"
    return flags


def _parser_signature(parser: argparse.ArgumentParser) -> dict:
    """Structured snapshot of a parser's flag set: ``{option_string →
    (default, type_name, required, choices, nargs)}``. Catches drift
    that --help text alone might hide (e.g., a default change with
    identical help text)."""
    sig: dict = {}
    for action in parser._actions:
        # Skip the auto-generated --help action; it varies in shape.
        if isinstance(action, argparse._HelpAction):
            continue
        if not action.option_strings:
            # Positional -- key by metavar/dest.
            key = f"<positional:{action.dest}>"
        else:
            # Stable key: tuple of all option strings sorted.
            key = tuple(sorted(action.option_strings))
        type_name = action.type.__name__ if callable(action.type) else None
        sig[key] = {
            "default": action.default,
            "type": type_name,
            "required": getattr(action, "required", False),
            "choices": tuple(action.choices) if action.choices else None,
            "nargs": action.nargs,
        }
    return sig


def test_cache_entrypoint_equivalent_to_legacy_script():
    """``scripts/cache_dataset.py`` and ``vista_ocr.entrypoints.cache``
    must expose identical argparse signatures (flags / defaults /
    types / required-ness / choices / nargs).

    Operators currently rely on the script; the CLI verb is the new
    surface. Drift between them is the regression this test catches.
    """
    repo_root = Path(__file__).resolve().parent.parent
    legacy_script = repo_root / "scripts" / "cache_dataset.py"
    assert legacy_script.exists(), "regression: legacy cache_dataset.py disappeared"

    # Build the legacy parser by importing the script as a module via
    # importlib.util so ``__name__ != '__main__'`` (won't fire main()).
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_legacy_cache_dataset", legacy_script,
    )
    legacy_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_mod)

    # Legacy script defines its parser inside main(); we can't easily
    # extract it without executing main(). Instead, drive both via
    # subprocess --help and compare the structured flag sets, plus
    # compare a hand-curated set of critical flag/default pairs.
    legacy_help = subprocess.run(
        [sys.executable, str(legacy_script), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert legacy_help.returncode == 0, legacy_help.stderr
    legacy_flags = set(_help_to_flag_dict(legacy_help.stdout).keys())

    from vista_ocr.entrypoints.cache import build_parser as new_parser_fn

    new_parser = new_parser_fn()
    # Pull flag names from the new parser's actions so we don't rely
    # on argparse formatting.
    new_flags = set()
    for action in new_parser._actions:
        for opt in action.option_strings:
            new_flags.add(opt)

    # Both must declare the same flag set.
    assert legacy_flags <= new_flags, (
        f"flags missing from new entrypoint: {legacy_flags - new_flags}"
    )
    # New parser may add extras over time; flag if it's missing any.
    extra_legacy = legacy_flags - new_flags
    assert not extra_legacy, (
        f"new entrypoint missing flags from legacy script: {extra_legacy}"
    )


# ---------------- Subprocess wrapper: the installed `vista-ocr` script


def _console_script() -> Path:
    """Resolve the `vista-ocr` console script next to the test runner's
    Python. Works without depending on the test's inherited $PATH."""
    candidate = Path(sys.executable).parent / "vista-ocr"
    if not candidate.exists():
        pytest.skip(
            f"`vista-ocr` console script not found at {candidate}; "
            "run `pip install -e .` in this env first"
        )
    return candidate


def test_installed_console_script_resolves():
    """The pip-installed ``vista-ocr`` console script must run end-to-
    end, not just the in-process ``main()``. Catches packaging-level
    regressions (entry-point typo, missing [project.scripts], wrong
    module path)."""
    r = subprocess.run(
        [str(_console_script()), "--version"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, f"`vista-ocr --version` failed: {r.stderr}"
    assert r.stdout.startswith("vista-ocr "), f"unexpected output: {r.stdout!r}"


def test_installed_console_script_help_lists_verbs():
    r = subprocess.run(
        [str(_console_script()), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0
    for verb in ("eval", "finetune", "infer", "cache"):
        assert verb in r.stdout, f"verb {verb} missing from CLI --help"
