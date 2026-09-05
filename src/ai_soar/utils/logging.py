"""Logging configuration for AI SOAR.

Call :func:`setup_logging` once at process start (scripts / API entrypoints).
Everywhere else, just use :func:`get_logger`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from ai_soar.config import LoggingConfig, Settings, get_settings


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Configure root logging from settings (console + optional file)."""
    cfg = config or get_settings().logging
    level = getattr(logging, cfg.level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if cfg.file is not None:
        log_path = Path(cfg.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(level=level, format=cfg.format, handlers=handlers, force=True)
    # Third-party libraries are noisy at INFO; keep them quiet unless debugging.
    for noisy in ("urllib3", "httpx", "matplotlib", "lightgbm"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


def get_logger(name: str) -> logging.Logger:
    """Module-level logger. Usage: ``log = get_logger(__name__)``."""
    return logging.getLogger(name)


def configure_from_settings(settings: Settings) -> None:
    """Alias kept for explicit call-sites that already hold a Settings object."""
    setup_logging(settings.logging)