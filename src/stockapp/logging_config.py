"""
Structured logging: JSON lines on stdout with fields ts, level, logger, event, duration_ms, ticker and user (the first
8 characters of the hashed user id). Extra fields whose names look secret (containing key, token, secret or password)
are redacted. Configure once per process with configure_logging().
"""

import json
import logging
import os
import sys
from datetime import UTC, datetime

REDACT_MARKERS = ("key", "token", "secret", "password")
REDACTED = "[REDACTED]"
USER_PREFIX = 8
_STANDARD = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime", "taskName"}
_configured = False


def _is_secret_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in REDACT_MARKERS)


def _jsonable(value: object) -> object:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Secret-looking extras are redacted; the user id is truncated to 8 characters."""

    def format(self, record: logging.LogRecord) -> str:
        extras = {k: v for k, v in vars(record).items() if k not in _STANDARD}
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": extras.pop("event", None) or record.getMessage(),
            "duration_ms": extras.pop("duration_ms", None),
            "ticker": extras.pop("ticker", None),
            "user": str(extras.pop("user"))[:USER_PREFIX] if extras.get("user") else None,
        }
        for name, value in extras.items():
            payload[name] = REDACTED if _is_secret_name(name) else _jsonable(value)
        if record.exc_info and record.exc_info[0] is not None:
            payload["error"] = record.exc_info[0].__name__  # type only: tracebacks can carry data
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str | None = None) -> None:
    """Send stockapp (and root) logs to stdout as JSON lines. Idempotent; the level defaults to LOG_LEVEL or INFO."""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level or os.environ.get("LOG_LEVEL", "INFO"))
    for noisy in ("yfinance", "peewee", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True
