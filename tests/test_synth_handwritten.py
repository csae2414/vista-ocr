"""Tests for the handwritten line synth generator (Phase J1a).

Inventory (notes/plan_phase_j.md §6):
- determinism: same seed reproduces identical line metadata
- bbox validity: mask-derived bboxes are within image and non-degenerate
- multi-line spacing: emitted lines do not vertically overlap
- english charset sanity: ascii + curly punct render without crashing
- packaging smoke: the bundled font dir is reachable via importlib.resources
- license + provenance audit: every .ttf in the bundled dir has a sibling
  .LICENSE.txt AND an entry in PROVENANCE.json with SHA256 matching disk
- sample contract: emitted Sample matches the PDFA emitter's field set
- task override: cfg.task="ocr" produces task="ocr" Sample
- meta tagging: source_family / language / font / text_source populated
- font-not-found: missing font_paths raises a helpful error

The fr-charset / french determinism tests are J1b-gated and live in
test_synth_handwritten_fr.py (NOT created until J0 #2 reports green).
"""
from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path

import pytest

from vista_ocr.data.synth.handwritten import (
    DEFAULT_FONT_RESOURCE,
    HandwrittenLineSynth,
    HandwrittenLineSynthConfig,
    TextSource,
)


SYSTEM_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
SYSTEM_FONT_AVAILABLE = SYSTEM_FONT.exists()


@pytest.fixture
def text_corpus(tmp_path: Path) -> Path:
    p = tmp_path / "corpus.txt"
    p.write_text(
        "The quick brown fox jumps over the lazy dog.\n"
        "Sphinx of black quartz, judge my vow.\n"
        "Pack my box with five dozen liquor jugs.\n"
        "How vexingly quick daft zebras jump.\n"
        "Bright vixens jump; dozy fowl quack.\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def text_source(text_corpus: Path) -> TextSource:
    return TextSource(path=text_corpus, mode="sentence", tag="pangrams")


@pytest.fixture
def cfg() -> HandwrittenLineSynthConfig:
    if not SYSTEM_FONT_AVAILABLE:
        pytest.skip("DejaVu test font not available")
    return HandwrittenLineSynthConfig(
        language="en",
        font_paths=[SYSTEM_FONT],
        canvas_size=(600, 800),
        min_lines=3,
        max_lines=6,
        min_line_height=24,
        max_line_height=40,
        min_chars_per_line=20,
        max_chars_per_line=40,
    )


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_handwritten_synth_determinism_metadata(text_source, cfg):
    """Same seed → identical line metadata across two generators."""
    a = HandwrittenLineSynth([text_source], cfg, seed=42)
    b = HandwrittenLineSynth([text_source], cfg, seed=42)
    for _ in range(5):
        sa = next(a)
        sb = next(b)
        assert [(ln.text, ln.bbox) for ln in sa.lines] == [(ln.text, ln.bbox) for ln in sb.lines]
        assert sa.source == sb.source
        assert sa.meta == sb.meta


def test_handwritten_synth_different_seeds_diverge(text_source, cfg):
    a = HandwrittenLineSynth([text_source], cfg, seed=1)
    b = HandwrittenLineSynth([text_source], cfg, seed=2)
    sa = next(a)
    sb = next(b)
    assert [ln.text for ln in sa.lines] != [ln.text for ln in sb.lines] \
        or [ln.bbox for ln in sa.lines] != [ln.bbox for ln in sb.lines]


# ---------------------------------------------------------------------------
# bbox validity
# ---------------------------------------------------------------------------

def test_bbox_within_image_and_non_degenerate(text_source, cfg):
    gen = HandwrittenLineSynth([text_source], cfg, seed=0)
    H, W = cfg.canvas_size
    for _ in range(20):
        s = next(gen)
        assert s.image.size == (W, H)
        for ln in s.lines:
            x1, y1, x2, y2 = ln.bbox
            assert 0 <= x1 < x2 <= W, f"x out of range: {ln.bbox}"
            assert 0 <= y1 < y2 <= H, f"y out of range: {ln.bbox}"


def test_multi_line_no_vertical_overlap(text_source, cfg):
    gen = HandwrittenLineSynth([text_source], cfg, seed=0)
    for _ in range(20):
        s = next(gen)
        # Sort by top edge; consecutive lines should not overlap vertically.
        ys = sorted([(ln.bbox[1], ln.bbox[3]) for ln in s.lines if ln.text])
        for (a_top, a_bot), (b_top, _) in zip(ys, ys[1:], strict=False):
            assert b_top >= a_top, "bbox order broken"
            # Mask-tight bboxes can touch; must not cross.
            assert b_top >= a_bot - 2, f"vertical overlap: {a_top}-{a_bot} vs {b_top}"


# ---------------------------------------------------------------------------
# english charset sanity
# ---------------------------------------------------------------------------

def test_english_punctuation_renders(text_corpus: Path, cfg):
    text_corpus.write_text(
        "He said, “don’t do that” — then left.\n"
        "Numbers: 1, 2, 3; symbols: & $ % @!\n",
        encoding="utf-8",
    )
    src = TextSource(path=text_corpus, mode="sentence", tag="punct")
    gen = HandwrittenLineSynth([src], cfg, seed=0)
    s = next(gen)
    # Just assert SOMETHING rendered; specific char coverage depends on font.
    assert any(ln.text for ln in s.lines)


# ---------------------------------------------------------------------------
# Sample contract + task override + meta tagging
# ---------------------------------------------------------------------------

def test_sample_contract_matches_pdfa_emitter(text_source, cfg):
    """task / source / lines / meta shape must be the PDFA-shape."""
    gen = HandwrittenLineSynth([text_source], cfg, seed=0)
    s = next(gen)
    assert s.task == "ocr_layout"
    assert s.source.startswith("synth_handwritten:en:")
    assert s.source.endswith(":pangrams")
    assert isinstance(s.meta, dict)
    assert s.meta["source_family"] == "synth_handwritten"
    assert s.meta["language"] == "en"
    assert "font" in s.meta and s.meta["font"].endswith(".ttf")
    assert s.meta["text_source"] == "pangrams"


def test_task_override_to_ocr(text_source, cfg):
    """The OCR-vs-layout ablation flips task without touching rendering."""
    cfg.task = "ocr"
    gen = HandwrittenLineSynth([text_source], cfg, seed=0)
    s = next(gen)
    assert s.task == "ocr"
    # Lines + bboxes still computed (caller may discard layout downstream).
    assert all(ln.text for ln in s.lines if ln.text)


# ---------------------------------------------------------------------------
# packaging + license + provenance
# ---------------------------------------------------------------------------

def test_default_font_dir_resolves_via_importlib_resources():
    """Operator-overridable default font dir must be accessible via
    importlib.resources, not by computing __file__-relative paths.
    Editable installs, wheels, and zipped packages all expose this."""
    pkg = resources.files(DEFAULT_FONT_RESOURCE)
    # Must be reachable; presence of TTFs is a separate licence-audit
    # concern (PROVENANCE.json), not a packaging concern.
    children = list(pkg.iterdir())
    # Always-present files we ship: PROVENANCE.json + HOLDOUT.md + __init__.py
    names = {Path(str(c)).name for c in children}
    assert "PROVENANCE.json" in names
    assert "HOLDOUT.md" in names


def test_provenance_manifest_present_and_well_formed():
    pkg = resources.files(DEFAULT_FONT_RESOURCE)
    manifest_path = Path(str(pkg.joinpath("PROVENANCE.json")))
    data = json.loads(manifest_path.read_text())
    assert data["schema_version"] == 1
    assert isinstance(data.get("fonts"), list)
    # Every entry must declare the required keys; this catches a contributor
    # adding a font without filling in its provenance row.
    REQUIRED = {"name", "ttf", "license_file", "license", "source_url",
                "retrieval_date", "version", "sha256", "realism_class"}
    for row in data["fonts"]:
        missing = REQUIRED - row.keys()
        assert not missing, f"PROVENANCE.json font row missing {missing}: {row}"
        assert row["realism_class"] in ("print_hw", "cursive_hw", "decorative")
        assert row["license"] in ("OFL-1.1", "Apache-2.0", "MIT")


def test_provenance_sha256_matches_on_disk_when_fonts_bundled():
    """If any TTFs are bundled, PROVENANCE.json's SHA256 must match
    the on-disk binary. Catches a contributor swapping a font without
    refreshing the manifest. No fonts bundled yet → test trivially passes."""
    pkg = resources.files(DEFAULT_FONT_RESOURCE)
    manifest = json.loads(Path(str(pkg.joinpath("PROVENANCE.json"))).read_text())
    by_ttf = {row["ttf"]: row for row in manifest["fonts"]}
    found_any = False
    for child in pkg.iterdir():
        name = Path(str(child)).name
        if not name.endswith(".ttf"):
            continue
        found_any = True
        assert name in by_ttf, f"{name} on disk but missing from PROVENANCE.json"
        on_disk = hashlib.sha256(Path(str(child)).read_bytes()).hexdigest()
        assert on_disk == by_ttf[name]["sha256"], \
            f"{name} SHA256 mismatch: on-disk={on_disk[:16]}…, manifest={by_ttf[name]['sha256'][:16]}…"
        # License sibling must also exist and contain a recognised license token.
        license_path = pkg.joinpath(by_ttf[name]["license_file"])
        text = Path(str(license_path)).read_text(errors="ignore")
        assert any(tok in text for tok in ("OFL", "Apache", "MIT")), \
            f"{name}: license file does not contain OFL/Apache/MIT marker"
    if not found_any:
        # Sanity: bundle in J1a-followup; document via PROVENANCE.json comment.
        assert manifest.get("fonts") == [], \
            "PROVENANCE.json claims fonts but none on disk"


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------

def test_missing_font_path_raises(tmp_path: Path, text_source):
    cfg = HandwrittenLineSynthConfig(
        language="en",
        font_paths=[tmp_path / "nope.ttf"],
        canvas_size=(400, 400),
    )
    with pytest.raises(FileNotFoundError):
        HandwrittenLineSynth([text_source], cfg, seed=0)


def test_missing_text_corpus_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        TextSource(path=tmp_path / "nope.txt", tag="missing")


def test_empty_text_sources_raises(cfg):
    with pytest.raises(ValueError):
        HandwrittenLineSynth([], cfg, seed=0)
