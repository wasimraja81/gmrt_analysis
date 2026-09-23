"""Unified logging setup for GWB pipeline stages.

Every stage gets the same convention: a per-run log file at DEBUG level, a
console stream at a configurable level (default INFO), and a
``<stage>_latest.log`` symlink so ``tail -f`` always follows the newest run
without knowing its run_id in advance. See
docs/user/GWB_USER_GUIDE.md#finding-out-what-a-run-did-logs for what this
means for you as the operator. Used by ``provenance.manifest.RunManifest``,
which shares its own ``run_id`` with the log filename so the two can be
cross-referenced by name.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"


def setup_stage_logger(
    stage: str,
    work_dir: Path,
    run_id: str,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a logger for one stage invocation.

    The logger is named uniquely per (stage, run_id) rather than per stage,
    so that running the same stage repeatedly in one process (as tests do)
    never accumulates handlers on a shared logger object.
    """
    log_dir = Path(work_dir) / "logs" / stage
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.log"

    logger = logging.getLogger(f"gwb_pipeline.{stage}.{run_id}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    _update_latest_symlink(log_dir, stage, log_path)

    return logger


def close_logger(logger: logging.Logger) -> None:
    """Close and detach every handler, releasing the log file handle."""
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _update_latest_symlink(log_dir: Path, stage: str, log_path: Path) -> None:
    latest_link = log_dir / f"{stage}_latest.log"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(log_path.name)
