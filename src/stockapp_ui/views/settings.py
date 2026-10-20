"""Settings page: reset shared settings and, when signed in, save / load / delete watchlists and settings."""

import streamlit as st

from stockapp.portfolio_input import WatchlistError, parse_ticker_text
from stockapp.userstore import UserDataError, UserStore
from stockapp_ui import auth, diagnostics, state


def _load_watchlist(store: UserStore, sub_hash: str, name: str) -> None:
    try:
        tickers = store.load_watchlist(sub_hash, name)
    except UserDataError as exc:
        st.session_state["settings_message"] = ("error", str(exc))
        return
    st.session_state["pf_source"] = "Type tickers"
    st.session_state["pf_tickers"] = ", ".join(tickers)
    st.session_state["settings_message"] = ("success", f"Loaded '{name}' into the Portfolio page.")


def _load_settings(store: UserStore, sub_hash: str) -> None:
    try:
        saved = store.load_settings(sub_hash)
    except UserDataError as exc:
        st.session_state["settings_message"] = ("error", str(exc))
        return
    if saved is None:
        st.session_state["settings_message"] = ("info", "No saved settings yet.")
        return
    state.apply_settings(saved)
    st.session_state["settings_message"] = ("success", "Saved settings loaded.")


def _reset_saved(store: UserStore, sub_hash: str) -> None:
    store.reset_settings(sub_hash)
    state.reset_to_defaults()
    st.session_state["settings_message"] = ("success", "Saved settings removed; defaults restored.")


def _delete_everything(store: UserStore, sub_hash: str) -> None:
    store.delete_user_data(sub_hash)
    st.session_state["confirm_delete"] = False
    st.session_state["settings_message"] = ("success", "All your saved data was deleted.")


def _saved_state(identity: auth.Identity) -> None:
    store = UserStore()
    sub_hash = identity.sub_hash

    st.markdown("**Watchlists**")
    with st.form("save_watchlist", clear_on_submit=False):
        name = st.text_input("Name", max_chars=40)
        text = st.text_area("Tickers (comma or space separated, up to 50)")
        if st.form_submit_button("Save watchlist"):
            try:
                store.save_watchlist(sub_hash, name.strip(), parse_ticker_text(text))
                st.success(f"Saved '{name.strip()}'.")
            except (WatchlistError, UserDataError) as exc:
                st.error(str(exc))
    names = store.list_watchlists(sub_hash)
    if names:
        chosen = st.selectbox("Saved watchlists", names, key="saved_watchlist")
        c1, c2 = st.columns(2)
        c1.button("Load into Portfolio", on_click=_load_watchlist, args=(store, sub_hash, chosen))
        if c2.button("Delete watchlist"):
            store.delete_watchlist(sub_hash, chosen)
            st.rerun()
    else:
        st.caption("No saved watchlists yet.")

    st.markdown("**Settings**")
    s1, s2, s3 = st.columns(3)
    if s1.button("Save current settings"):
        try:
            store.save_settings(sub_hash, state.current_settings())
            st.success("Settings saved.")
        except UserDataError as exc:
            st.error(str(exc))
    s2.button("Load saved settings", on_click=_load_settings, args=(store, sub_hash))
    s3.button("Reset saved settings", on_click=_reset_saved, args=(store, sub_hash))

    st.markdown("**Delete my data**")
    st.caption("Removes every saved watchlist and setting for your account. This can't be undone.")
    confirmed = st.checkbox("I understand: delete all my saved data", key="confirm_delete")
    st.button(
        "Delete my data", type="primary", disabled=not confirmed, on_click=_delete_everything, args=(store, sub_hash)
    )


def render() -> None:
    st.subheader("Settings")
    st.caption("Ticker, history, base currency and Kelly settings live in the sidebar and are shared by every page.")
    st.button("Reset sidebar settings to defaults", on_click=state.reset_to_defaults)
    message = st.session_state.pop("settings_message", None)
    if message:
        kind, text = message
        {"success": st.success, "error": st.error}.get(kind, st.info)(text)

    st.divider()
    identity = auth.current_identity()
    problem = auth.saved_state_problem()
    if identity is not None:
        _saved_state(identity)
    elif problem:
        st.warning(problem)
    elif auth.auth_configured():
        st.info("Sign in from the sidebar to save watchlists and settings.")
    else:
        st.caption("Saved watchlists and settings need sign-in, which isn't configured on this server.")

    if diagnostics.enabled():
        with st.expander("Diagnostics"):
            events = diagnostics.session_events()
            hits = sum(e.cache == "hit" for e in events)
            st.metric("App version", diagnostics.app_version())
            st.metric("Data-layer cache hit rate (this session)", f"{hits / len(events):.0%}" if events else "n/a")
            st.caption(
                "Counts fetches that reached the SQLite data layer; repeat views served from the in-memory cache "
                "don't reach it."
            )
            if events:
                st.dataframe(
                    [{"ticker": e.ticker, "cache": e.cache, "ms": e.duration_ms} for e in reversed(events[-20:])]
                )
