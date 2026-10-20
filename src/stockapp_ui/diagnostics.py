"""
Per-session diagnostics for the Settings page (shown when DEBUG_UI=1): a logging handler keeps each Streamlit
session's recent data-fetch events, tagged by the session that ran them.
"""

import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

from streamlit.runtime.scriptrunner import get_script_run_ctx

MAX_EVENTS = 200


@dataclass(frozen=True)
class FetchEvent:
    ticker: str
    cache: str
    duration_ms: float


class SessionFetchHandler(logging.Handler):
    """Records data_fetch log events per Streamlit session id (events outside a session are ignored)."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.events: dict[str, deque[FetchEvent]] = defaultdict(lambda: deque(maxlen=MAX_EVENTS))

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "event", None) != "data_fetch":
            return
        ctx = get_script_run_ctx(suppress_warning=True)
        if ctx is None:
            return
        event = FetchEvent(
            ticker=str(getattr(record, "ticker", "")),
            cache=str(getattr(record, "cache", "")),
            duration_ms=float(getattr(record, "duration_ms", 0.0)),
        )
        self.events[ctx.session_id].append(event)


_session_users: dict[str, str] = {}


class SessionUserFilter(logging.Filter):
    """Adds the current session's hashed user id (if signed in) to records that don't carry one."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "user", None):
            ctx = get_script_run_ctx(suppress_warning=True)
            if ctx is not None and ctx.session_id in _session_users:
                record.user = _session_users[ctx.session_id]
        return True


def set_session_user(sub_hash: str | None) -> None:
    """Record (or clear) the signed-in user's hash for the current session's log lines."""
    ctx = get_script_run_ctx(suppress_warning=True)
    if ctx is None:
        return
    if sub_hash:
        _session_users[ctx.session_id] = sub_hash
    else:
        _session_users.pop(ctx.session_id, None)


_handler: SessionFetchHandler | None = None


def install() -> None:
    """Attach the fetch recorder and the user filter to the root logger's handlers, once per process."""
    global _handler
    if _handler is None:
        _handler = SessionFetchHandler()
        root = logging.getLogger()
        root.addHandler(_handler)
        for handler in root.handlers:
            handler.addFilter(SessionUserFilter())


def session_events() -> list[FetchEvent]:
    ctx = get_script_run_ctx(suppress_warning=True)
    if _handler is None or ctx is None:
        return []
    return list(_handler.events.get(ctx.session_id, ()))


def enabled() -> bool:
    return os.environ.get("DEBUG_UI") == "1"


def app_version() -> str:
    try:
        return version("stockapp")
    except PackageNotFoundError:
        return "unknown"
