"""Central logging setup shared by every entry point.

The pre-refactor repo had exactly one logger, in ``train.py``, and it appended a fresh
``FileHandler`` on every call -- so calling it twice duplicated every subsequent line.
Everything else used ``print()`` (``utils.py`` had 11 calls, 8 of them commented-out debug
prints, which is a debugging strategy rather than an accident). ``rich`` is already a
dependency, so console output gets colourised through a ``RichHandler`` when available.

Call :func:`get_logger` from any module. It is idempotent per name.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

__all__ = ["get_logger", "set_level"]

_CONFIGURED: dict[str, logging.Logger] = {}
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def _console_handler() -> logging.Handler:
    """A Rich handler when importable, otherwise a plain stream handler."""
    try:
        from rich.logging import RichHandler

        return RichHandler(
            rich_tracebacks=True,
            show_path=False,
            omit_repeated_times=False,
            console=None,
        )
    except ImportError:  # pragma: no cover - rich is a hard dependency, this is the fallback
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        return handler


def _file_handler(logfile: Path) -> logging.Handler:
    handler = logging.FileHandler(logfile, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))
    return handler


def get_logger(
    name: str = "sbb",
    level: str | int = "INFO",
    logfile: str | Path | None = None,
) -> logging.Logger:
    """Return a configured logger, creating and caching it on first call.

    Args:
        name: Logger name; ``"sbb"`` is the package default and sub-namespaces such as
            ``"sbb.train"`` inherit its handlers.
        level: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ... or a ``logging`` level constant.
        logfile: Path to a log file. If a directory is given, a timestamped
            ``sbb_YYYYmmdd_HHMMSS.log`` is created inside it. ``None`` means console only.

    Returns:
        A logger with console (and optional file) handlers attached exactly once. Repeated
        calls with the same ``name`` return the cached instance and never duplicate handlers.
    """
    cached = _CONFIGURED.get(name)
    if cached is not None and logfile is None:
        cached.setLevel(level)
        return cached
    if cached is not None:
        # Same name, new logfile: reuse the logger and add the file handler if it is new.
        logger = cached
    else:
        logger = logging.getLogger(name)
        logger.propagate = False
        logger.addHandler(_console_handler())
        _CONFIGURED[name] = logger

    logger.setLevel(level)

    if logfile is not None:
        path = Path(logfile)
        if path.is_dir() or (path.suffix == "" and not path.exists()):
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path / f"sbb_{datetime.now():%Y%m%d_%H%M%S}.log"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

        resolved = str(path.resolve())
        if not any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == resolved
            for h in logger.handlers
        ):
            logger.addHandler(_file_handler(path))

    return logger


def set_level(level: str | int, name: str = "sbb") -> None:
    """Change the level of an already-configured logger."""
    get_logger(name).setLevel(level)
