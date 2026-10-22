import json

import pandas as pd
import pytest

from stockapp import scan
from stockapp.scan import ScanRow, changed_actions, render_html, send_alert

MALICIOUS = "<script>alert('x')</script>"


def test_html_escapes_every_value():
    rows = [ScanRow(MALICIOUS, "HOLD", 0.25, 0.5, False, "<b>normal</b>", note=f'"><img src=x onerror={MALICIOUS}>')]
    page = render_html(rows, "2026-10-20T00:00:00+00:00")
    assert "<script>" not in page and "<img" not in page and "<b>" not in page
    assert "&lt;script&gt;" in page
    assert "Not investment advice" in page


def test_changed_actions_compares_with_previous_scan():
    previous = {"MSFT": {"action": "HOLD"}, "SPY": {"action": "BUY / ADD"}}
    rows = [ScanRow("MSFT", "HOLD", 0, 0.5, False, None), ScanRow("SPY", "HOLD", 0, 0.5, False, None),
            ScanRow("NEW", "HOLD", 0, 0.5, False, None)]  # fmt: skip
    assert [r.ticker for r in changed_actions(previous, rows)] == ["SPY", "NEW"]


class FakeSMTP:
    instances: list["FakeSMTP"] = []

    def __init__(self, host, port, timeout):
        self.host, self.port, self.timeout = host, port, timeout
        self.calls: list[str] = []
        self.messages = []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, context):
        self.calls.append("starttls")

    def login(self, user, password):
        self.calls.append("login")

    def send_message(self, message):
        self.calls.append("send")
        self.messages.append(message)


ENV = {"SMTP_HOST": "smtp.example.com", "SMTP_PORT": "587", "SMTP_USER": "me@example.com", "SMTP_PASS": "pw",
       "ALERT_TO": "you@example.com"}  # fmt: skip


def test_email_uses_starttls_before_login(monkeypatch):
    FakeSMTP.instances.clear()
    monkeypatch.setattr(scan.smtplib, "SMTP", FakeSMTP)
    send_alert([ScanRow("SPY", "SELL / TRIM", 0.0, 0.4, True, "high")], ENV)
    smtp = FakeSMTP.instances[0]
    assert (smtp.host, smtp.port) == ("smtp.example.com", 587)
    assert smtp.calls == ["starttls", "login", "send"]
    body = smtp.messages[0].get_content()
    assert "SPY: SELL / TRIM" in body and "pw" not in body


def test_main_writes_reports_and_emails_only_changes(monkeypatch, tmp_path):
    FakeSMTP.instances.clear()
    watchlist = tmp_path / "watchlist.csv"
    watchlist.write_text("ticker,weight\nMSFT,0.3\nSPY,0\n")
    out = tmp_path / "scan.html"
    out.with_suffix(".json").write_text(
        json.dumps({"actions": {"MSFT": {"action": "HOLD"}, "SPY": {"action": "HOLD"}}})
    )
    actions = {"MSFT": "HOLD", "SPY": "BUY / ADD"}
    monkeypatch.setattr(scan, "fetch_rf_annual_pct", lambda years: pd.Series(dtype=float))
    monkeypatch.setattr(
        scan, "scan_ticker", lambda ticker, current, *args: ScanRow(ticker, actions[ticker], current, 0.5, False, "low")
    )
    monkeypatch.setattr(scan.smtplib, "SMTP", FakeSMTP)
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)

    assert scan.main(["--watchlist", str(watchlist), "--out", str(out), "--email"]) == 0

    saved = json.loads(out.with_suffix(".json").read_text())["actions"]
    assert saved["MSFT"]["target"] == 0.3 and saved["SPY"]["action"] == "BUY / ADD"
    assert "<table" in out.read_text()
    body = FakeSMTP.instances[0].messages[0].get_content()
    assert "SPY: BUY / ADD" in body and "MSFT" not in body


def test_main_rejects_invalid_watchlist(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("ticker,notes\nMSFT,x\n")
    assert scan.main(["--watchlist", str(bad), "--out", str(tmp_path / "r.html")]) == 2
    assert "Unexpected column" in capsys.readouterr().err


def test_email_skipped_without_credentials(monkeypatch, tmp_path):
    watchlist = tmp_path / "w.csv"
    watchlist.write_text("ticker\nMSFT\n")
    monkeypatch.setattr(scan, "fetch_rf_annual_pct", lambda years: pd.Series(dtype=float))
    monkeypatch.setattr(scan, "scan_ticker", lambda ticker, *args: ScanRow(ticker, "HOLD", 0.0, 0.5, False, None))
    for key in ENV:
        monkeypatch.delenv(key, raising=False)
    assert scan.main(["--watchlist", str(watchlist), "--out", str(tmp_path / "r.html"), "--email"]) == 3


@pytest.mark.parametrize("value", [None, 0.25])
def test_html_formats_missing_and_numeric_values(value):
    page = render_html([ScanRow("MSFT", "ERROR", value, value, None, None, "boom")], "now")
    assert ("25.00%" in page) == (value is not None)
