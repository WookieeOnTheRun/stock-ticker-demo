import socket

import curl_cffi.curl
import pytest


def _network_blocked(*args, **kwargs):
    raise RuntimeError("Network access is disabled in tests")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any network connection. yfinance uses curl_cffi (libcurl), which bypasses Python sockets."""
    monkeypatch.setattr(socket.socket, "connect", _network_blocked)
    monkeypatch.setattr(curl_cffi.curl.Curl, "perform", _network_blocked)
