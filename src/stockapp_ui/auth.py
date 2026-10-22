"""
Optional sign-in with Streamlit's native OIDC auth (st.login / st.user / st.logout, Streamlit 1.66 + Authlib).
Without a complete [auth] section in secrets the app runs anonymously and no sign-in button is shown. Users are
identified only by an HMAC (keyed with app_salt) of the ID token's issuer and subject claims; no raw claim is kept.
"""

import hashlib
import hmac
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from stockapp.userstore import UserStore

logger = logging.getLogger(__name__)

MIGRATED_KEY = "_auth_key_migrated"
"""Session-state key holding the user key whose legacy data has already been checked this session."""

# Keys Streamlit 1.66 requires (streamlit/auth_util.py, validate_auth_credentials)
AUTH_KEYS = ("redirect_uri", "cookie_secret")
PROVIDER_KEYS = ("client_id", "client_secret", "server_metadata_url")


def _secrets_section(name: str) -> Mapping[str, object] | None:
    """The [name] table from st.secrets, or None if there are no secrets or no such table."""
    try:
        value = st.secrets.get(name)
    except (StreamlitSecretNotFoundError, FileNotFoundError):
        return None
    return value if isinstance(value, Mapping) else None


def _secret(name: str) -> str | None:
    """A top-level secret from st.secrets, falling back to the environment."""
    try:
        value = st.secrets.get(name)
    except (StreamlitSecretNotFoundError, FileNotFoundError):
        value = None
    return str(value) if value else os.environ.get(name) or None


def provider_name() -> str | None:
    """
    The OIDC provider to sign in with: None for a default provider (client keys directly under [auth]), the first
    named [auth.<provider>] section otherwise. Raises nothing; returns None when auth isn't configured.
    """
    auth = _secrets_section("auth")
    if auth is None or all(k in auth for k in PROVIDER_KEYS):
        return None
    named = [k for k, v in auth.items() if isinstance(v, Mapping) and all(p in v for p in PROVIDER_KEYS)]
    return named[0] if named else None


def auth_configured() -> bool:
    """True only if [auth] has redirect_uri ending in /oauth2callback, cookie_secret, and one complete provider."""
    auth = _secrets_section("auth")
    if auth is None or not all(auth.get(k) for k in AUTH_KEYS):
        return False
    if not str(auth["redirect_uri"]).endswith("/oauth2callback"):
        return False
    return all(k in auth for k in PROVIDER_KEYS) or provider_name() is not None


def _logged_in() -> bool:
    """st.user.is_logged_in, treating a missing attribute as signed out (Streamlit only adds it once [auth] exists)."""
    return bool(st.user.to_dict().get("is_logged_in", False))


@dataclass(frozen=True)
class Identity:
    """The signed-in user, known only by a keyed hash of the provider's issuer and subject claims."""

    sub_hash: str
    """hash_subject(app_salt, iss, sub): the only user identifier stored or logged (first 8 characters)."""


def hash_subject(app_salt: str, issuer: str, sub: str) -> str:
    """
    HMAC-SHA256 keyed with app_salt over the JSON array [issuer, sub], as hex: the user key UserStore uses. OpenID
    Connect only guarantees that sub is unique within one issuer, so the issuer is part of the key and users of two
    different providers never share one. JSON keeps the two fields unambiguous (no other split gives the same message).
    Without app_salt the key can't be linked back to an account.
    """
    message = json.dumps([issuer, sub]).encode()
    return hmac.new(app_salt.encode(), message, hashlib.sha256).hexdigest()


def legacy_hash_subject(app_salt: str, sub: str) -> str:
    """The previous user key, sha256(app_salt + sub) with no issuer; used only to migrate data saved under it."""
    return hashlib.sha256((app_salt + sub).encode()).hexdigest()


def _migrate_legacy_key(app_salt: str, sub: str, sub_hash: str) -> None:
    """
    Move any data saved under the user's legacy key to their current key, checked once per session. This assumes
    the legacy data was saved through the provider configured now: the legacy key has no issuer, so after a provider
    switch it could belong to a different person with the same sub (see README, "Secrets and sign-in"). A failure is
    logged and retried on the next run; data is never merged into a key that already has some (UserStore.migrate_user).
    """
    if st.session_state.get(MIGRATED_KEY) == sub_hash:
        return
    try:
        moved = UserStore().migrate_user(legacy_hash_subject(app_salt, sub), sub_hash)
    except Exception:
        logger.warning(
            "user key migration failed", exc_info=True, extra={"event": "user_key_migration_failed", "user": sub_hash}
        )
        return
    st.session_state[MIGRATED_KEY] = sub_hash
    if moved:
        logger.info("user key migrated", extra={"event": "user_key_migrated", "user": sub_hash})


def current_identity() -> Identity | None:
    """
    The signed-in user's hashed identity, or None when anonymous, auth is off, no app_salt is configured, or the
    provider sent no sub or iss claim. Also moves the user's data from the legacy key on first use in a session.
    """
    if not auth_configured() or not _logged_in():
        return None
    claims = st.user.to_dict()
    sub, issuer = claims.get("sub"), claims.get("iss")
    salt = _secret("app_salt")
    if not sub or not issuer or not salt:
        missing = "no 'sub' claim" if not sub else "no 'iss' claim" if not issuer else "no app_salt"
        logger.warning("Saved state disabled: %s", missing)
        return None
    identity = Identity(hash_subject(salt, str(issuer), str(sub)))
    _migrate_legacy_key(salt, str(sub), identity.sub_hash)
    return identity


def saved_state_problem() -> str | None:
    """Why saved state is unavailable for a signed-in user (None if it is available or the user isn't signed in)."""
    if not auth_configured() or not _logged_in():
        return None
    claims = st.user.to_dict()
    if not claims.get("sub"):
        return "Your identity provider did not send a 'sub' claim, so settings can't be saved."
    if not claims.get("iss"):
        return "Your identity provider did not send an 'iss' claim, so settings can't be saved."
    if not _secret("app_salt"):
        return "Saving is disabled: the administrator hasn't set app_salt in secrets."
    return None


def render_sidebar_controls() -> None:
    """Sign-in / sign-out buttons in the sidebar, shown only when auth is configured."""
    if not auth_configured():
        return
    with st.sidebar:
        if _logged_in():
            st.caption("Signed in.")
            st.button("Sign out", on_click=st.logout, key="sign_out")
        else:
            provider = provider_name()
            st.button("Sign in to save watchlists and settings", key="sign_in", on_click=lambda: st.login(provider))
