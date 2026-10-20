"""
Watchlist scanner: runs the advice rules for every ticker in a watchlist CSV through the cached data layer and
writes an HTML report (every value escaped) plus a JSON file of actions. Optionally emails the tickers whose action
changed since the previous JSON, over SMTP with STARTTLS, using credentials from the environment only.

    python -m stockapp.scan --watchlist watchlist.csv --out report.html [--email]
"""

import argparse
import html
import json
import logging
import os
import smtplib
import ssl
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path

import pandas as pd

from stockapp.advice import BANNER, AdviceInputs, decide, kelly_inputs
from stockapp.fx import convert_prices, fx_ticker, major_currency
from stockapp.model.probability import walk_forward_model
from stockapp.portfolio_input import MAX_UPLOAD_BYTES, WatchlistError, parse_watchlist_csv
from stockapp.regimes import vol_regimes
from stockapp.repository import load_bars, load_currency
from stockapp.riskfree import align_rf, daily_rf, fetch_rf_annual_pct

logger = logging.getLogger("stockapp.scan")
SMTP_ENV = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "ALERT_TO")
SMTP_TIMEOUT_S = 30
BASE_CURRENCY = "USD"


@dataclass(frozen=True)
class ScanRow:
    ticker: str
    action: str
    target: float | None
    p: float | None
    model_has_skill: bool | None
    regime: str | None
    note: str = ""


def scan_ticker(
    ticker: str,
    current: float,
    rf_annual_pct: pd.Series,
    horizon: int,
    years: int,
    lookback: int,
    k_fraction: float = 0.5,
    prior_sd_annual: float = 0.10,
    max_dd: float = 0.25,
) -> ScanRow:
    """Advice for one ticker: USD prices, the walk-forward model (p = 0.5 without skill), Kelly on `lookback` days."""
    native = load_bars(ticker, years)["adj_close"]
    major, divisor = major_currency(load_currency(ticker) or BASE_CURRENCY)
    rate = load_bars(fx_ticker(major, BASE_CURRENCY), years)["close"] if major != BASE_CURRENCY else None
    prices = convert_prices(native, divisor, rate)
    rf = align_rf(daily_rf(rf_annual_pct), pd.DatetimeIndex(prices.index))
    prices = prices.loc[rf.index]
    report = walk_forward_model(prices, rf, horizon)
    p = report.latest_probability if report.has_skill else 0.5
    kelly = kelly_inputs(prices.iloc[-(lookback + 1) :], rf, k_fraction, prior_sd_annual, max_dd)
    regimes = vol_regimes(prices.pct_change()).dropna()
    regime = str(regimes.iloc[-1]) if len(regimes) else None
    advice = decide(AdviceInputs(p, kelly.kelly_p5, kelly.target, regime, kelly.dd_kelly, current))
    return ScanRow(ticker, advice.action, round(advice.new_position, 4), round(p, 4), report.has_skill, regime)


def render_html(rows: list[ScanRow], generated: str) -> str:
    """A standalone HTML table of scan results; every value is HTML-escaped."""

    def cell(value: object) -> str:
        text = "" if value is None else f"{value:.2%}" if isinstance(value, float) else str(value)
        return f"<td>{html.escape(text)}</td>"

    header = "".join(
        f"<th>{html.escape(h)}</th>" for h in ("Ticker", "Action", "Target", "p", "Model skill", "Regime", "Note")
    )
    body = "\n".join(
        "<tr>"
        + "".join(cell(v) for v in (r.ticker, r.action, r.target, r.p, r.model_has_skill, r.regime, r.note))
        + "</tr>"
        for r in rows
    )
    return (
        "<!doctype html>\n<html><head><meta charset='utf-8'><title>Watchlist scan</title></head><body>\n"
        f"<p>{html.escape(BANNER)}</p>\n<p>Generated {html.escape(generated)}</p>\n"
        f"<table border='1'>\n<tr>{header}</tr>\n{body}\n</table>\n</body></html>\n"
    )


def changed_actions(previous: dict[str, dict[str, object]], rows: list[ScanRow]) -> list[ScanRow]:
    """Rows whose action differs from the previous scan (new tickers count as changed)."""
    return [r for r in rows if previous.get(r.ticker, {}).get("action") != r.action]


def send_alert(changes: list[ScanRow], env: dict[str, str]) -> None:
    """Email the changed actions via SMTP with STARTTLS. Credentials come from `env`; nothing secret is logged."""
    message = EmailMessage()
    message["Subject"] = f"Watchlist scan: {len(changes)} action change(s)"
    message["From"] = env["SMTP_USER"]
    message["To"] = env["ALERT_TO"]
    lines = [f"{r.ticker}: {r.action} (target {r.target if r.target is not None else 'n/a'})" for r in changes]
    message.set_content("\n".join([*lines, "", BANNER]))
    with smtplib.SMTP(env["SMTP_HOST"], int(env["SMTP_PORT"]), timeout=SMTP_TIMEOUT_S) as smtp:
        smtp.starttls(context=ssl.create_default_context())
        smtp.login(env["SMTP_USER"], env["SMTP_PASS"])
        smtp.send_message(message)
    logger.info("Alert sent for %d ticker(s)", len(changes))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m stockapp.scan", description=__doc__.split("\n\n")[0])
    parser.add_argument("--watchlist", required=True, type=Path, help="CSV with a 'ticker' column (optional weight)")
    parser.add_argument("--out", required=True, type=Path, help="HTML report path; a .json file is written alongside")
    parser.add_argument("--email", action="store_true", help="email tickers whose action changed (SMTP_* env vars)")
    parser.add_argument("--horizon", type=int, choices=(5, 21), default=21)
    parser.add_argument("--years", type=int, default=10, help="history to load per ticker")
    parser.add_argument("--lookback", type=int, default=252, help="trading days for the Kelly estimate")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        if args.watchlist.stat().st_size > MAX_UPLOAD_BYTES:
            raise WatchlistError("The watchlist file is larger than 1 MB.")
        watchlist = parse_watchlist_csv(args.watchlist.read_bytes())
    except (OSError, WatchlistError) as exc:
        print(f"Watchlist error: {exc}", file=sys.stderr)
        return 2
    json_path = args.out.with_suffix(".json")
    previous: dict[str, dict[str, object]] = {}
    if json_path.exists():
        previous = json.loads(json_path.read_text()).get("actions", {})

    rf_annual_pct = fetch_rf_annual_pct(args.years)
    current = watchlist.weights or {}
    rows = []
    for ticker in watchlist.tickers:
        try:
            rows.append(
                scan_ticker(ticker, current.get(ticker, 0.0), rf_annual_pct, args.horizon, args.years, args.lookback)
            )
        except Exception as exc:
            logger.warning("%s failed: %s", ticker, type(exc).__name__)
            rows.append(ScanRow(ticker, "ERROR", None, None, None, None, f"{type(exc).__name__}: {exc}"))
        logger.info("%s: %s", ticker, rows[-1].action)

    generated = datetime.now(UTC).isoformat(timespec="seconds")
    args.out.write_text(render_html(rows, generated))
    json_path.write_text(
        json.dumps({"generated_utc": generated, "actions": {r.ticker: asdict(r) for r in rows}}, indent=2)
    )
    print(f"Wrote {args.out} and {json_path}")

    if args.email:
        changes = changed_actions(previous, rows)
        missing = [k for k in SMTP_ENV if not os.environ.get(k)]
        if missing:
            print(f"Email skipped: set {', '.join(missing)}", file=sys.stderr)
            return 3
        if changes:
            send_alert(changes, {k: os.environ[k] for k in SMTP_ENV})
        else:
            print("No action changes; no email sent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
