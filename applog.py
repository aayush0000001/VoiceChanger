"""
A small rotating log file in %LOCALAPPDATA%\\V-COMMS (never inside the project folder), plus an exception hook so a
bug in a slot is logged and reported instead of silently ending the app (PyQt6 aborts the process on an unhandled
exception unless sys.excepthook is replaced). The real-time audio callback never logs.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Callable, List, Optional

from appinfo import APP_NAME

LOG_FILE = "vcomms.log"
MAX_BYTES = 256 * 1024
LOGGER_NAME = "vcomms"


def log_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / APP_NAME / LOG_FILE


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup_logging() -> logging.Logger:
    """Idempotent. Falls back to a null handler when the log folder cannot be written."""
    logger = get_logger()
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=1, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    except OSError:
        logger.addHandler(logging.NullHandler())
    return logger


def tail(max_lines: int = 50) -> List[str]:
    """The last ``max_lines`` lines of the log, or an empty list when there is no log."""
    try:
        with open(log_path(), "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(size - 64 * 1024, 0))
            data = f.read()
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    return lines[-max_lines:]


def install_excepthooks(on_error: Optional[Callable[[str], None]] = None) -> None:
    """Log every uncaught exception (main thread, Qt slots and plain threads) and tell ``on_error`` about it."""
    logger = get_logger()

    def hook(exc_type, exc, tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        logger.error("Unhandled exception\n%s", text.rstrip())
        try:
            sys.__stderr__.write(text)
        except Exception:
            pass
        if on_error is not None:
            try:
                on_error(f"{exc_type.__name__}: {exc}")
            except Exception:
                pass

    sys.excepthook = hook
    threading.excepthook = lambda args: hook(args.exc_type, args.exc_value, args.exc_traceback)
