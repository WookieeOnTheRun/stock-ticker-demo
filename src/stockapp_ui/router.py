"""Page registry: which pages exist, and which tiers they need."""

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class PageSpec:
    path: str
    """Page script, relative to src/app.py."""
    title: str
    icon: str
    url_path: str
    requires: str
    """Module that must be importable for the page to be registered (the tier that provides it)."""


PAGE_SPECS = [
    PageSpec("app_pages/research.py", "Research", ":material/query_stats:", "research", "stockapp.kelly"),
    PageSpec(
        "app_pages/fundamentals.py", "Fundamentals", ":material/description:", "fundamentals", "stockapp.fundamentals"
    ),
    PageSpec("app_pages/portfolio.py", "Portfolio", ":material/pie_chart:", "portfolio", "stockapp.optimize"),
    PageSpec("app_pages/backtest.py", "Backtest & Signals", ":material/timeline:", "backtest", "stockapp.backtest"),
    PageSpec("app_pages/settings.py", "Settings", ":material/settings:", "settings", "stockapp_ui.state"),
]


def available_pages() -> list[PageSpec]:
    """Pages whose providing tier is installed; e.g. Fundamentals is absent until Tier 4 adds stockapp.fundamentals."""
    return [spec for spec in PAGE_SPECS if importlib.util.find_spec(spec.requires) is not None]
