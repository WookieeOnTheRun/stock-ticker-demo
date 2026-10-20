import socket

import curl_cffi.curl
import pytest

from stockapp import config


def _network_blocked(*args, **kwargs):
    raise RuntimeError("Network access is disabled in tests")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any network connection. yfinance uses curl_cffi (libcurl), which bypasses Python sockets."""
    monkeypatch.setattr(socket.socket, "connect", _network_blocked)
    monkeypatch.setattr(curl_cffi.curl.Curl, "perform", _network_blocked)


@pytest.fixture(autouse=True)
def isolated_data_dir(monkeypatch, tmp_path):
    """Point the SQLite cache at a per-test data/ directory so tests never touch the real cache."""
    data_dir = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "DB_PATH", data_dir / "market.sqlite")
    return data_dir
