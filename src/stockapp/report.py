"""
Self-contained HTML research report for one ticker. Rendered with Jinja2 autoescaping, so every user-controlled or
third-party string (ticker, table cells, an LLM brief, headlines) is escaped; the only raw HTML is Plotly's own
figure markup. plotly.js is embedded once, inline, so the file renders offline. To get a PDF, print it from a browser.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment
from markupsafe import Markup

from stockapp.advice import BANNER

logger = logging.getLogger(__name__)

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ ticker }} research report</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; color: #222; }
  table { border-collapse: collapse; margin: 0.5rem 0 1.5rem; }
  th, td { border: 1px solid #ccc; padding: 0.3rem 0.6rem; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  .banner { background: #fff4e5; border: 1px solid #f0c36d; padding: 0.6rem 1rem; }
  .meta { color: #666; }
</style>
</head>
<body>
<h1>{{ ticker }} research report</h1>
<p class="meta">Generated {{ generated }} · data through {{ data_through }} · prices in {{ currency }}</p>
<p class="banner">{{ disclaimer }}</p>
{% for table in tables %}
<h2>{{ table.title }}</h2>
{% if table.rows %}
<table>
<tr>{% for column in table.columns %}<th>{{ column }}</th>{% endfor %}</tr>
{% for row in table.rows %}<tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>{% endfor %}
</table>
{% else %}
<p>{{ table.note }}</p>
{% endif %}
{% endfor %}
{% for chart in charts %}
<h2>{{ chart.title }}</h2>
{{ chart.html }}
{% endfor %}
<h2>AI-generated summary</h2>
{% if brief %}<p><strong>AI-generated summary — not investment advice.</strong></p><pre>{{ brief }}</pre>
{% else %}<p>No AI-generated brief is available for this report.</p>{% endif %}
<p class="banner">{{ disclaimer }}</p>
</body>
</html>
"""

_TEMPLATE = Environment(autoescape=True).from_string(TEMPLATE)  # compiled once per process, not per report


@dataclass
class ReportTable:
    """One table in the report; cells are already formatted strings."""

    title: str
    columns: list[str]
    rows: list[list[str]]
    note: str = ""
    """Shown instead of the table when there are no rows (e.g. a data source that isn't available)."""


@dataclass
class ReportData:
    """Everything build_report renders for one ticker."""

    ticker: str
    data_through: str
    currency: str
    tables: list[ReportTable] = field(default_factory=list)
    figures: dict[str, go.Figure] = field(default_factory=dict)
    brief: str | None = None


def build_report(data: ReportData, generated: datetime | None = None) -> str:
    """Render the report as one HTML string with plotly.js embedded inline in the first chart only."""
    started = time.perf_counter()
    charts = []
    for i, (title, fig) in enumerate(data.figures.items()):
        markup = pio.to_html(fig, include_plotlyjs=i == 0, full_html=False)
        charts.append({"title": title, "html": Markup(markup)})  # noqa: S704 - Plotly's own figure HTML
    html = _TEMPLATE.render(
        ticker=data.ticker,
        generated=(generated or datetime.now(UTC)).isoformat(timespec="seconds"),
        data_through=data.data_through,
        currency=data.currency,
        disclaimer=BANNER,
        tables=data.tables,
        charts=charts,
        brief=data.brief,
    )
    logger.info(
        "report generated",
        extra={
            "event": "report_generated",
            "ticker": data.ticker,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )
    return html
