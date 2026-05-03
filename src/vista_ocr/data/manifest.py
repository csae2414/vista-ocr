"""JSONL manifest -- the canonical input format for ``vista-ocr eval``,
``vista-ocr finetune``, and ``vista-ocr infer``.

One line per document. Required fields: ``image`` (path relative to
the manifest's directory, or absolute), ``ref`` (whitespace-joined
ground-truth text). Optional fields: ``bboxes`` (list of
``[x1, y1, x2, y2, text]`` for layout-aware metrics), ``task`` (one of
``ocr`` / ``ocr_layout`` / ``region_ocr`` / ``find_it``; default
``ocr_layout``), ``query_text`` / ``query_bbox`` (for find_it /
region_ocr respectively), ``version`` (default 1).

Schema policy (v1): unknown ``version`` values are a hard error. The
file format will only declare a ``v2`` once we have a concrete
breaking change to motivate it; until then the strict reject keeps us
from painting ourselves into a forward-compat corner.

Example::

    {"image": "img/001.jpg", "ref": "Hello world"}
    {"image": "img/002.jpg", "ref": "Foo bar",
     "bboxes": [[10, 20, 100, 50, "Foo"]]}
    {"image": "img/003.jpg", "ref": "Baz", "task": "ocr"}
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from PIL import Image

from vista_ocr.data.types import Sample, TaskName
from vista_ocr.tokenizer.tokenizer import Line

SUPPORTED_VERSIONS: tuple[int, ...] = (1,)
VALID_TASKS: tuple[TaskName, ...] = ("ocr", "ocr_layout", "region_ocr", "find_it")


def _err(line_no: int, msg: str) -> ValueError:
    return ValueError(f"manifest line {line_no}: {msg}")


def _parse_record(rec: dict[str, Any], line_no: int) -> tuple[str, dict]:
    """Validate a single manifest record and return ``(image_rel, parsed_kwargs)``.

    Returns the validated parts but does not yet open the image -- the
    caller resolves the path against the manifest's directory.
    """
    if not isinstance(rec, dict):
        raise _err(line_no, f"record must be a JSON object, got {type(rec).__name__}")

    version = rec.get("version", 1)
    # Strict type check: ``1.0`` (float) and ``"1"`` (str) silently
    # equal ``1`` in Python's ``in`` operator / numeric comparison,
    # which is the wrong semantic for a schema-version field. Reject
    # anything that's not a plain int (excluding bool, which subclasses
    # int).
    if not isinstance(version, int) or isinstance(version, bool):
        raise _err(
            line_no,
            f"'version' must be an integer; got {version!r} ({type(version).__name__})",
        )
    if version not in SUPPORTED_VERSIONS:
        raise _err(
            line_no,
            f"unsupported manifest version {version!r}; "
            f"this build supports {SUPPORTED_VERSIONS}",
        )

    if "image" not in rec or not isinstance(rec["image"], str) or not rec["image"]:
        raise _err(line_no, "missing or empty required field 'image'")
    if "ref" not in rec or not isinstance(rec["ref"], str):
        raise _err(line_no, "missing or non-string required field 'ref'")

    task = rec.get("task", "ocr_layout")
    if task not in VALID_TASKS:
        raise _err(
            line_no,
            f"task={task!r} not in {VALID_TASKS}",
        )

    bboxes_raw = rec.get("bboxes")
    lines: list[Line] = []
    if bboxes_raw is not None:
        if not isinstance(bboxes_raw, list):
            raise _err(line_no, "'bboxes' must be a list")
        for j, item in enumerate(bboxes_raw):
            if not isinstance(item, list) or len(item) != 5:
                raise _err(
                    line_no,
                    f"bboxes[{j}] must be [x1,y1,x2,y2,text]; got {item!r}",
                )
            x1, y1, x2, y2, text = item
            if not all(isinstance(v, (int, float)) for v in (x1, y1, x2, y2)):
                raise _err(line_no, f"bboxes[{j}] coords must be numeric")
            if not isinstance(text, str):
                raise _err(line_no, f"bboxes[{j}] text must be a string")
            lines.append(Line(text=text, bbox=(int(x1), int(y1), int(x2), int(y2))))

    # If the caller didn't supply per-line bboxes, fall back to a single
    # whole-page line carrying the ref text. The model's loss path
    # tolerates this; the bbox is left as a degenerate (0,0,0,0).
    if not lines:
        lines.append(Line(text=rec["ref"], bbox=(0, 0, 0, 0)))

    parsed = {
        "task": task,
        "lines": lines,
        "query_text": rec.get("query_text"),
        "query_bbox": (
            tuple(rec["query_bbox"]) if rec.get("query_bbox") is not None else None
        ),
    }
    return rec["image"], parsed


def iter_manifest_records(path: Path | str) -> Iterator[tuple[Sample, dict[str, Any]]]:
    """Yield ``(Sample, raw_dict)`` per manifest line.

    Use this when a caller needs to know which optional fields were
    present in the source record (e.g., the eval verb dispatches
    detection metrics on whether ``bboxes`` was supplied vs
    synthesised by :func:`iter_manifest`'s fallback). The Sample
    object alone can't distinguish "operator omitted bboxes" from
    "operator passed bboxes that happened to project to (0,0,0,0)".
    """
    path = Path(path)
    base = path.parent
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as e:
                raise _err(line_no, f"invalid JSON: {e}") from e
            image_rel, parsed = _parse_record(rec, line_no)
            image_path = Path(image_rel)
            if not image_path.is_absolute():
                image_path = base / image_path
            if not image_path.exists():
                raise _err(line_no, f"image not found: {image_path}")
            with Image.open(image_path) as img:
                img = img.convert("L")
                img.load()
            sample = Sample(
                image=img,
                lines=parsed["lines"],
                task=parsed["task"],
                query_text=parsed["query_text"],
                query_bbox=parsed["query_bbox"],
                source=f"manifest:{path.name}:{line_no}",
            )
            yield sample, rec


def iter_manifest(path: Path | str) -> Iterator[Sample]:
    """Yield :class:`Sample` per manifest line.

    ``image`` paths are resolved relative to the manifest's directory
    so manifests are portable. Absolute paths are honoured as-is.
    Validation is strict: any malformed record halts iteration with
    ``ValueError`` (do not eat errors -- a noisy schema violation is
    more useful than a silently-skipped doc).

    Callers that also need access to the raw record (e.g. to detect
    which optional fields were present in the source JSONL) should
    use :func:`iter_manifest_records` instead.
    """
    for sample, _rec in iter_manifest_records(path):
        yield sample


def write_manifest(records: Iterable[dict], path: Path | str) -> int:
    """Write ``records`` (validated dicts in v1 schema) to ``path`` as
    JSONL; returns the number of lines written.

    The dict shape mirrors the parsed input: required ``image`` and
    ``ref``; optional ``bboxes``, ``task``, ``query_text``,
    ``query_bbox``, ``version``. Each record is validated before being
    written so a malformed call fails fast (and the file is not
    half-written).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(records, 1):
            # Validate record shape before serialising. We deliberately
            # don't open images here -- write_manifest is a producer-
            # side helper and shouldn't require the images to exist on
            # disk yet (test fixtures may build a manifest before
            # rendering).
            _validate_for_write(rec, i)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def _validate_for_write(rec: dict, line_no: int) -> None:
    if not isinstance(rec, dict):
        raise _err(line_no, "record must be a dict")
    if "image" not in rec or not isinstance(rec["image"], str) or not rec["image"]:
        raise _err(line_no, "missing or empty required field 'image'")
    if "ref" not in rec or not isinstance(rec["ref"], str):
        raise _err(line_no, "missing or non-string required field 'ref'")
    task = rec.get("task", "ocr_layout")
    if task not in VALID_TASKS:
        raise _err(line_no, f"task={task!r} not in {VALID_TASKS}")
    version = rec.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool):
        raise _err(
            line_no,
            f"'version' must be an integer; got {version!r}",
        )
    if version not in SUPPORTED_VERSIONS:
        raise _err(line_no, f"unsupported version {version!r}")
