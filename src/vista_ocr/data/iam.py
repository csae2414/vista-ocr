"""IAM Handwriting Database loader.

Paper Table 3 reports VISTA-OCR's WER on IAM at 10.14. We load IAM in the
classical full-page layout (Aachen splits): one image per writing form,
with line-level transcriptions + bounding boxes parsed from
``ascii/lines.txt`` and ``xml/*.xml``.

The IAM dataset is licence-restricted; users register at the FKI Bern
website and download manually. We don't fetch it for them. This loader
expects a directory layout produced by ``prepare_iam.py`` (which the user
runs once after extracting the IAM zips):

    iam_root/
      forms/
        a01-000u.png
        a01-000x.png
        ...
      lines/
        a01-000u/
          a01-000u-00.png        # (line crops; we use full forms by default)
          ...
      xml/
        a01-000u.xml             # line-level coordinates + transcription
      splits/
        train.txt
        val.txt
        test.txt

Each XML has::

    <line id="..." text="..." asx="..." asy="..." asw="..." ash="...">

We translate that into our :class:`Sample` shape with line-level bboxes.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.tokenizer import Line

LOG = logging.getLogger(__name__)


@dataclass
class IamConfig:
    root: Path
    split: str = "train"                  # train / val / test
    drop_zero_lines: bool = True


def _read_split(cfg: IamConfig) -> list[str]:
    p = cfg.root / "splits" / f"{cfg.split}.txt"
    if not p.exists():
        raise FileNotFoundError(f"IAM split file missing: {p}")
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def _parse_form_xml(xml_path: Path) -> list[Line]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines: list[Line] = []
    for line_elem in root.iter("line"):
        text = line_elem.attrib.get("text", "").strip()
        if not text:
            continue
        # IAM stores the ASCII (line-level) bbox as separate attrs.
        try:
            x = int(line_elem.attrib["asx"])
            y = int(line_elem.attrib["asy"])
            w = int(line_elem.attrib["asw"])
            h = int(line_elem.attrib["ash"])
        except KeyError:
            # Some forms encode word-level only; fall back to enclosing the
            # word bboxes inside the line.
            words = list(line_elem.iter("word"))
            if not words:
                continue
            xs, ys, ws, hs = [], [], [], []
            for w_elem in words:
                try:
                    xs.append(int(w_elem.attrib["x"]))
                    ys.append(int(w_elem.attrib["y"]))
                    ws.append(int(w_elem.attrib["width"]))
                    hs.append(int(w_elem.attrib["height"]))
                except KeyError:
                    continue
            if not xs:
                continue
            x = min(xs)
            y = min(ys)
            w = max(xs[i] + ws[i] for i in range(len(xs))) - x
            h = max(ys[i] + hs[i] for i in range(len(ys))) - y
        lines.append(Line(text=text, bbox=(x, y, x + w, y + h)))
    return lines


def iter_iam(cfg: IamConfig) -> Iterator[Sample]:
    """Yield one :class:`Sample` per form image in ``cfg.split``."""
    form_ids = _read_split(cfg)
    forms_dir = cfg.root / "forms"
    xml_dir = cfg.root / "xml"
    for form_id in form_ids:
        png = forms_dir / f"{form_id}.png"
        xml = xml_dir / f"{form_id}.xml"
        if not png.exists() or not xml.exists():
            LOG.warning("IAM form missing: %s", form_id)
            continue
        try:
            lines = _parse_form_xml(xml)
        except ET.ParseError as e:
            LOG.warning("XML parse failed for %s: %s", form_id, e)
            continue
        if cfg.drop_zero_lines and not lines:
            continue
        img = Image.open(png).convert("L")
        yield Sample(image=img, lines=lines, task="ocr_layout",
                     source=f"iam:{cfg.split}:{form_id}")
