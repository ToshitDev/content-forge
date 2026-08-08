"""Configures logging once for the whole app: console + rotating file.

Modules just do `logger = logging.getLogger(__name__)` and log normally —
that works fine with zero setup. This module is what makes those log
records actually go somewhere useful; it's called once, at each real
entry point (pipeline.py's __main__, app.py), not by library code.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOGS_DIR / "contentforge.log"
MAX_BYTES = 1_000_000  # rotate once a log file reaches ~1MB
BACKUP_COUNT = 3

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a console handler and a rotating file handler to the root logger.

    Safe to call more than once — only configures on the first call.
    That matters because Streamlit reruns the whole script on every
    interaction; without this guard, app.py would stack up a fresh pair
    of handlers (and duplicate every log line) on every rerun.
    """
    global _configured
    if _configured:
        return
    _configured = True

    LOGS_DIR.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
