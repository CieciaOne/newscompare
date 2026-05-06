"""Reduce noisy HTTP client logs; keep app-level INFO useful."""

from __future__ import annotations

import logging


def quiet_http_loggers() -> None:
    """httpx/httpcore log every request at INFO; silence unless debugging."""
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
