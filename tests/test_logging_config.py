"""Logging-config tests."""
from __future__ import annotations

import logging

from vista_ocr.logging_config import setup_logging


def test_setup_logging_clears_existing_handlers(tmp_path):
    log = tmp_path / "x.log"
    setup_logging(level="DEBUG", log_file=log)
    n_after = len(logging.getLogger().handlers)
    setup_logging(level="DEBUG", log_file=log)   # called again — must not stack
    assert len(logging.getLogger().handlers) == n_after


def test_setup_logging_writes_to_file(tmp_path):
    log = tmp_path / "y.log"
    setup_logging(level="INFO", log_file=log)
    logging.getLogger("vista_ocr.test").info("hello world")
    for h in logging.getLogger().handlers:
        h.flush()
    assert "hello world" in log.read_text()
