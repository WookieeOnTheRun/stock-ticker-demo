import hashlib
import hmac
import json
from types import SimpleNamespace

import pytest

from stockapp.userstore import UserStore
from stockapp_ui import auth

SALT = "pepper"
ISSUER = "https://issuer.example"
SECRETS = {
    "app_salt": SALT,
    "auth": {
        "redirect_uri": "http://localhost:8501/oauth2callback",
        "cookie_secret": "cookie",
        "client_id": "id",
        "client_secret": "secret",
        "server_metadata_url": f"{ISSUER}/.well-known/openid-configuration",
    },
}


class _FakeUser:
    """Stands in for st.user: to_dict() returns the ID-token claims plus is_logged_in."""

    def __init__(self, claims):
        self._claims = claims

    def to_dict(self):
        return dict(self._claims)


@pytest.fixture
def signed_in(monkeypatch):
    """Replace auth's `st` with a signed-in fake carrying the given claims; returns the fake."""

    def install(**claims):
        fake = SimpleNamespace(
            secrets=dict(SECRETS), user=_FakeUser({"is_logged_in": True, **claims}), session_state={}
        )
        monkeypatch.setattr(auth, "st", fake)
        return fake

    return install


def test_hash_subject_is_hmac_of_issuer_and_sub():
    expected = hmac.new(SALT.encode(), json.dumps([ISSUER, "123"]).encode(), hashlib.sha256).hexdigest()
    assert auth.hash_subject(SALT, ISSUER, "123") == expected
    assert auth.hash_subject(SALT, ISSUER, "123") != auth.legacy_hash_subject(SALT, "123")


def test_same_sub_at_different_issuers_gets_different_keys():
    assert auth.hash_subject(SALT, "https://a.example", "123") != auth.hash_subject(SALT, "https://b.example", "123")


def test_issuer_and_sub_boundaries_are_unambiguous():
    assert auth.hash_subject(SALT, "ab", "c") != auth.hash_subject(SALT, "a", "bc")


def test_sign_in_moves_legacy_data_to_the_new_key(signed_in):
    store = UserStore()
    legacy = auth.legacy_hash_subject(SALT, "123")
    store.save_watchlist(legacy, "Core", ["MSFT", "AAPL"])
    store.save_settings(legacy, {"ticker": "SPY"})
    signed_in(sub="123", iss=ISSUER)

    identity = auth.current_identity()

    assert identity is not None
    assert identity.sub_hash == auth.hash_subject(SALT, ISSUER, "123")
    assert store.load_watchlist(identity.sub_hash, "Core") == ["MSFT", "AAPL"]
    assert store.load_settings(identity.sub_hash) == {"ticker": "SPY"}
    assert store.list_watchlists(legacy) == [] and store.load_settings(legacy) is None


def test_legacy_key_is_checked_once_per_session(signed_in, monkeypatch):
    signed_in(sub="123", iss=ISSUER)
    first = auth.current_identity()

    def must_not_run(self, old_hash, new_hash):
        raise AssertionError("migration ran twice in one session")

    monkeypatch.setattr(UserStore, "migrate_user", must_not_run)
    assert auth.current_identity() == first


def test_failed_migration_still_signs_in_and_is_retried(signed_in, monkeypatch):
    fake = signed_in(sub="123", iss=ISSUER)

    def broken(self, old_hash, new_hash):
        raise OSError("disk unavailable")

    monkeypatch.setattr(UserStore, "migrate_user", broken)
    identity = auth.current_identity()
    assert identity is not None
    assert auth.MIGRATED_KEY not in fake.session_state  # not marked done, so the next run tries again


def test_missing_issuer_disables_saved_state(signed_in):
    signed_in(sub="123")
    assert auth.current_identity() is None
    assert "'iss'" in auth.saved_state_problem()
