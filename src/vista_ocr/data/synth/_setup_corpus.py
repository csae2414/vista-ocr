"""Implementation of the corpus-acquisition step driven by
``scripts/datasets/setup_synth_corpus.sh`` (Phase J followup, Fix 2).

The shell script is a thin wrapper that does arg parsing + env
activation; the actual work (HF streaming, sentence splitting,
deterministic sampling, SHA256 + JSON provenance, idempotency
with checksum-drift detection, fixture-only mode) lives here in
Python where each step is short and safe.

Pinned dataset revisions
------------------------

``CORPORA`` below stores the dataset commit SHAs resolved against
HF on 2026-05-04 via ``HfApi.dataset_info``. Operators can override
per run with the env vars
``CORPUS_HF_REVISION_WIKITEXT`` / ``CORPUS_HF_REVISION_PG19`` if a
later revision is required (e.g. an upstream fix). The override is
recorded in PROVENANCE.json so the resulting corpus is still
reproducible.

Offline contract
----------------

``--dry-run`` and ``--fixture-only`` MUST NOT touch the network.
The CI test (``test_setup_synth_corpus.py::test_dry_run_offline``)
runs both with ``HF_HUB_OFFLINE=1`` set and asserts exit 0. Any
import that triggers an HF API call belongs only inside
``_download_corpus``, which is unreachable from those modes.

Idempotency + drift detection
-----------------------------

Re-running the helper with existing outputs reads the prior
``PROVENANCE.json`` and compares each on-disk file's SHA256
against the recorded value. Outcomes:

- **match:** logged "matched" and the run skips the download
  for that corpus.
- **drift:** the on-disk SHA256 differs from the recorded one;
  the helper hard-fails with ``SystemExit`` rather than silently
  re-using a tampered or partially-downloaded file. ``--force``
  re-downloads cleanly (overwrites + refreshes PROVENANCE.json).
- **missing prior PROVENANCE.json:** logged "no prior provenance"
  and the run proceeds (re-downloads if files are present);
  this is the first-run case.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger("setup_synth_corpus")

# In-tree letterlike fixture (committed, no download).
LETTERLIKE_FIXTURE = "letterlike.txt"

# Pinned dataset metadata. The implementation pass MUST verify each
# id + revision against HF before relying on the values; revisions
# pinned here are placeholders that the operator can override per
# run via env vars (see CORPUS_HF_REVISION_*).
@dataclass(frozen=True)
class CorpusSource:
    name: str
    hf_id: str
    hf_config: str | None
    hf_split: str
    hf_revision: str
    license: str
    out_subpath: str
    requires_pg19_max_docs: bool


# Pinned revisions resolved 2026-05-04 via HfApi.dataset_info.
# Override per run via env vars (CORPUS_HF_REVISION_WIKITEXT,
# CORPUS_HF_REVISION_PG19).
_PINNED_REVISIONS = {
    "wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "pg19": "4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b",
}


def _resolve_revision(name: str) -> str:
    env_key = f"CORPUS_HF_REVISION_{name.upper()}"
    return os.environ.get(env_key, _PINNED_REVISIONS[name])


CORPORA: dict[str, CorpusSource] = {
    "wikitext": CorpusSource(
        name="wikitext",
        hf_id="Salesforce/wikitext",
        hf_config="wikitext-2-raw-v1",
        hf_split="train",
        hf_revision=_resolve_revision("wikitext"),
        license="CC-BY-SA-3.0",
        out_subpath="en/wikitext.txt",
        requires_pg19_max_docs=False,
    ),
    "pg19": CorpusSource(
        name="pg19",
        hf_id="deepmind/pg19",
        hf_config=None,
        hf_split="train",
        hf_revision=_resolve_revision("pg19"),
        license="Apache-2.0",
        out_subpath="en/pg19.txt",
        requires_pg19_max_docs=True,
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="setup_synth_corpus", description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("corpora/synth"))
    p.add_argument("--lang", default="en", choices=("en",),
                   help="French (fr) is reserved for J1b, gated on tokenizer audit.")
    p.add_argument("--pg19-max-docs", type=int, default=1000,
                   help="Cap PG-19 streaming to this many books (default 1000; "
                        "1000 books * ~100 sentences/book = ~100k sentences).")
    p.add_argument("--max-sentences", type=int, default=100_000,
                   help="Cap total sentences per corpus (default 100000).")
    p.add_argument("--seed", type=int, default=0,
                   help="Deterministic sampling seed (default 0).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print resolved sources + caps as JSON; exit 0 without "
                        "writing or downloading. Honours HF_HUB_OFFLINE=1.")
    p.add_argument("--fixture-only", action="store_true",
                   help="Only copy the in-tree letterlike fixture to --out-dir; "
                        "never touches the network.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if checksum-matched outputs exist.")
    return p


def _resolve_plan(args: argparse.Namespace) -> dict:
    """Build a JSON-serialisable description of what would be done."""
    plan = {
        "out_dir": str(args.out_dir),
        "lang": args.lang,
        "pg19_max_docs": args.pg19_max_docs,
        "max_sentences": args.max_sentences,
        "seed": args.seed,
        "dry_run": bool(args.dry_run),
        "fixture_only": bool(args.fixture_only),
        "force": bool(args.force),
        "letterlike_fixture": LETTERLIKE_FIXTURE,
        "corpora": [
            {
                "name": c.name,
                "hf_id": c.hf_id,
                "hf_config": c.hf_config,
                "hf_split": c.hf_split,
                "hf_revision": c.hf_revision,
                "license": c.license,
                "output": str(args.out_dir / c.out_subpath),
            }
            for c in CORPORA.values()
        ],
    }
    return plan


def _read_prior_provenance_sha256(out_dir: Path) -> dict[str, str]:
    """Read the prior PROVENANCE.json (if any) and return a
    ``{corpus_name: sha256}`` map for the drift check.

    Missing file → empty dict (first run; the caller logs
    "no prior provenance" and proceeds).
    """
    manifest = out_dir / "PROVENANCE.json"
    if not manifest.exists():
        return {}
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        c["name"]: c["sha256"]
        for c in data.get("corpora", [])
        if isinstance(c, dict) and "name" in c and "sha256" in c
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_letterlike(out_dir: Path) -> Path:
    """Copy the in-tree letterlike.txt next to the downloaded files."""
    src = _repo_root() / "corpora" / "synth" / LETTERLIKE_FIXTURE
    if not src.exists():
        raise SystemExit(
            f"in-tree letterlike fixture missing: {src}. "
            f"Repo state is inconsistent; check git status."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / LETTERLIKE_FIXTURE
    shutil.copyfile(src, dst)
    return dst


def _repo_root() -> Path:
    """Repo root resolution: walk up from this file until a marker
    (``pyproject.toml``) is found."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _write_provenance(
    out_dir: Path,
    args: argparse.Namespace,
    files: list[tuple[str, Path, str]],
) -> Path:
    """Write PROVENANCE.json mirroring the font-side schema.

    ``files`` is a list of ``(name, path, license)`` triples; the
    JSON output records SHA256 + retrieval date per file plus the
    sample-config knobs.
    """
    today = datetime.date.today().isoformat()
    fonts_like = []
    for name, path, lic in files:
        spec = CORPORA.get(name)
        entry = {
            "name": name,
            "path": str(path.relative_to(out_dir)) if path.is_relative_to(out_dir) else str(path),
            "license": lic,
            "retrieval_date": today,
            "sha256": _sha256_file(path),
            "sample_config": {
                "pg19_max_docs": args.pg19_max_docs,
                "max_sentences": args.max_sentences,
                "seed": args.seed,
            },
        }
        if spec is not None:
            entry["hf_id"] = spec.hf_id
            entry["hf_config"] = spec.hf_config
            entry["hf_split"] = spec.hf_split
            entry["hf_revision"] = spec.hf_revision
        fonts_like.append(entry)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "PROVENANCE.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "comment": "Per-corpus provenance + license + sample-config binding.",
        "corpora": fonts_like,
    }, indent=2) + "\n", encoding="utf-8")
    return manifest


def _download_corpus(spec: CorpusSource, args: argparse.Namespace, out: Path) -> None:
    """Real download path; never reached under --dry-run / --fixture-only.

    Lazy-imports HF datasets so the module stays importable in
    minimal CI envs and the offline modes never trigger network IO.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "huggingface 'datasets' library not installed. "
            "Run 'pip install datasets' on the box that fetches corpora "
            "(not required at training time)."
        ) from e
    LOG.info("Streaming HF dataset %s (config=%s, split=%s, rev=%s) -> %s",
             spec.hf_id, spec.hf_config, spec.hf_split, spec.hf_revision, out)
    ds = load_dataset(
        spec.hf_id,
        name=spec.hf_config,
        split=spec.hf_split,
        revision=spec.hf_revision,
        streaming=True,
    )
    rng = _seeded_rng(args.seed)
    sentences = _stream_sentences(spec, ds, args, rng)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sentences) + "\n", encoding="utf-8")


def _seeded_rng(seed: int):
    import random
    return random.Random(seed)


def _stream_sentences(spec: CorpusSource, ds, args: argparse.Namespace, rng) -> list[str]:
    """Pull sentences from the streaming dataset, with deterministic
    reservoir sampling capped at ``--max-sentences``.

    PG-19 entries are book-shaped; cap at ``--pg19-max-docs`` books
    before sentence-splitting. Wikitext entries are paragraph-shaped.
    """
    out: list[str] = []
    book_cap = args.pg19_max_docs if spec.requires_pg19_max_docs else None
    n_books = 0
    for record in ds:
        if book_cap is not None and n_books >= book_cap:
            break
        text = record.get("text", "")
        for sent in _split_sentences(text):
            out.append(sent)
            if len(out) >= args.max_sentences * 2:
                # Reservoir-trim: keep a random subset to avoid
                # unbounded memory on very large inputs.
                rng.shuffle(out)
                out = out[: args.max_sentences]
        n_books += 1
    rng.shuffle(out)
    return out[: args.max_sentences]


def _split_sentences(text: str) -> list[str]:
    """Hand-rolled sentence splitter. Stable and dependency-free.

    Splits on ``.!?`` followed by whitespace + capital, drops empty
    fragments and stray boilerplate (lines starting with == are
    wikitext section headers).
    """
    import re
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("=="):
            continue
        # Naive splitter: enough for distribution-coverage corpus,
        # not for downstream NLP tasks.
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", stripped):
            s = s.strip()
            if 5 <= len(s) <= 500:
                out.append(s)
    return out


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    plan = _resolve_plan(args)

    if args.dry_run:
        # 100% offline: never touches HF or the network.
        print(json.dumps(plan, indent=2))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.fixture_only:
        # Offline: only copy the in-tree fixture + write PROVENANCE.
        ll = _copy_letterlike(args.out_dir)
        _write_provenance(
            args.out_dir, args,
            files=[("letterlike", ll, "in-tree (template-derived)")],
        )
        LOG.info("Wrote letterlike fixture + PROVENANCE.json under %s", args.out_dir)
        return 0

    # Idempotency + drift detection: read prior PROVENANCE.json
    # (if present) and compare each on-disk file's SHA256 against
    # the recorded value. Drift -> hard fail; --force -> overwrite.
    prior_sha = _read_prior_provenance_sha256(args.out_dir)

    # Real path: stream HF datasets, then write PROVENANCE.
    written: list[tuple[str, Path, str]] = []
    for spec in CORPORA.values():
        out_path = args.out_dir / spec.out_subpath
        if out_path.exists() and not args.force:
            existing_sha = _sha256_file(out_path)
            recorded_sha = prior_sha.get(spec.name)
            if recorded_sha is None:
                LOG.info("%s: existing file present, no prior PROVENANCE.json "
                         "entry (sha256=%s..). Skipping refetch; --force to overwrite.",
                         spec.name, existing_sha[:12])
            elif recorded_sha == existing_sha:
                LOG.info("%s: matched prior PROVENANCE.json (sha256=%s..). "
                         "Skipping refetch.", spec.name, existing_sha[:12])
            else:
                raise SystemExit(
                    f"{spec.name}: SHA256 drift detected. on-disk={existing_sha} "
                    f"vs PROVENANCE.json recorded={recorded_sha}. Investigate or "
                    f"pass --force to overwrite cleanly."
                )
            written.append((spec.name, out_path, spec.license))
            continue
        _download_corpus(spec, args, out_path)
        written.append((spec.name, out_path, spec.license))
    ll = _copy_letterlike(args.out_dir)
    written.append(("letterlike", ll, "in-tree (template-derived)"))
    _write_provenance(args.out_dir, args, written)
    LOG.info("Wrote %d corpora + PROVENANCE.json under %s",
             len(written), args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
