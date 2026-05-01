"""IAM loader tests using fabricated XML + PNG fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from vista_ocr.data.iam import IamConfig, _parse_form_xml, iter_iam


@pytest.fixture
def iam_root(tmp_path: Path) -> Path:
    root = tmp_path / "iam"
    (root / "forms").mkdir(parents=True)
    (root / "xml").mkdir(parents=True)
    (root / "splits").mkdir(parents=True)

    Image.new("L", (200, 100), 255).save(root / "forms" / "form0.png")
    Image.new("L", (200, 100), 255).save(root / "forms" / "form1.png")
    (root / "xml" / "form0.xml").write_text(
        '<form><line id="01a" text="hello world" '
        'asx="10" asy="20" asw="100" ash="30"/>'
        '<line id="01b" text="line two" '
        'asx="10" asy="60" asw="80" ash="20"/></form>'
    )
    (root / "xml" / "form1.xml").write_text(
        '<form><line id="00" text=""></line></form>'
    )
    (root / "splits" / "train.txt").write_text("form0\nform1\n")
    return root


def test_parse_form_xml_extracts_line_bboxes(iam_root: Path):
    lines = _parse_form_xml(iam_root / "xml" / "form0.xml")
    assert len(lines) == 2
    assert lines[0].text == "hello world"
    assert lines[0].bbox == (10, 20, 110, 50)
    assert lines[1].bbox == (10, 60, 90, 80)


def test_iter_iam_drops_empty_forms(iam_root: Path):
    samples = list(iter_iam(IamConfig(root=iam_root, split="train", drop_zero_lines=True)))
    sources = [s.source for s in samples]
    assert any("form0" in s for s in sources)
    assert not any("form1" in s for s in sources)


def test_iter_iam_yields_pil_images(iam_root: Path):
    samples = list(iter_iam(IamConfig(root=iam_root, split="train")))
    assert samples
    s = samples[0]
    assert hasattr(s.image, "size")
    assert s.image.size == (200, 100)


def test_iter_iam_missing_split_file_raises(tmp_path: Path):
    (tmp_path / "splits").mkdir()
    with pytest.raises(FileNotFoundError):
        list(iter_iam(IamConfig(root=tmp_path, split="train")))
