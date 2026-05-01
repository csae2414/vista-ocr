"""Sphinx configuration for vista-ocr."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the package importable for autodoc.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

project = "vista-ocr"
author = "vista-ocr contributors"
copyright = "2026, vista-ocr contributors"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# Mock heavy deps so docs can build without GPU/data on contributors' boxes.
autodoc_mock_imports = ["webdataset", "wandb", "datasets", "albumentations", "faker"]
