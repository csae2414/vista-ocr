"""Centralised logging setup.

Library code uses ``logging.getLogger(__name__)``. CLI scripts call
:func:`setup_logging` once at startup to configure handlers and level.

Example:
    >>> from vista_ocr.logging_config import setup_logging
    >>> setup_logging(level="INFO", log_file="logs/run.log")
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    fmt: str = _FMT,
) -> None:
    """Configure the root logger with a stream handler (stderr) and an
    optional rotating file handler.

    :param level: Logging level (string name or integer constant).
    :param log_file: Optional path to a log file; rotated at 10 MB × 5 backups.
    :param fmt: Format string for log records.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    formatter = logging.Formatter(fmt)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        rotating.setFormatter(formatter)
        root.addHandler(rotating)
