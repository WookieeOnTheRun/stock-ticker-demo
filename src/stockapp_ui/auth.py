"""
Optional sign-in with Streamlit's native OIDC auth (st.login / st.user / st.logout, Streamlit 1.66 + Authlib).
Without a complete [auth] section in secrets the app runs anonymously and no sign-in button is shown. Users are
identified only by sha256(app_salt + sub), where sub is the provider's stable subject claim; no raw claim is kept.
"""

import hashlib
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

logger = logging.getLogger(__name__)

# Keys Streamlit 1.66 requires (streamlit/auth_util.py, validate_auth_credentials)
AUTH_KEYS = ("redirect_uri", "cookie_secret")
PROVIDER_KEYS = ("client_id", "client_secret", "server_metadata_url")


def _secrets_section(name: str) -> Mapping[str, object] | None:
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
    sub_hash: str
    """sha256(app_salt + sub) hex digest: the only user identifier stored or logged (first 8 characters)."""


def hash_subject(app_salt: str, sub: str) -> str:
    return hashlib.sha256((app_salt + sub).encode()).hexdigest()


def current_identity() -> Identity | None:
    """The signed-in user's hashed identity, or None when anonymous, auth is off, or no app_salt is configured."""
    if not auth_configured() or not _logged_in():
        return None
    sub = st.user.to_dict().get("sub")
    salt = _secret("app_salt")
    if not sub or not salt:
        logger.warning("Saved state disabled: %s", "no 'sub' claim from the provider" if not sub else "no app_salt")
        return None
    return Identity(hash_subject(salt, str(sub)))


def saved_state_problem() -> str | None:
    """Why saved state is unavailable for a signed-in user (None if it is available or the user isn't signed in)."""
    if not auth_configured() or not _logged_in():
        return None
    if not st.user.to_dict().get("sub"):
        return "Your identity provider did not send a 'sub' claim, so settings can't be saved."
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
