"""Runtime configuration, overridable with environment variables. Holds no secrets."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
"""Every on-disk cache must live inside this directory."""

DB_PATH = Path(os.environ.get("STOCKAPP_DB_PATH", "data/market.sqlite"))
"""SQLite cache location; a relative path is resolved against PROJECT_ROOT."""

FRESHNESS_HOURS = float(os.environ.get("STOCKAPP_FRESHNESS_HOURS", "6"))
"""Cached bars younger than this are served without contacting the provider."""
