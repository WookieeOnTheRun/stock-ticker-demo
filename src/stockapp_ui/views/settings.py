"""Settings page: reset shared settings (saved watchlists and settings are added with sign-in)."""

import streamlit as st

from stockapp_ui import state


def render() -> None:
    st.subheader("Settings")
    st.caption("Ticker, history, base currency and Kelly settings live in the sidebar and are shared by every page.")
    if st.button("Reset settings to defaults"):
        for key, value in state.DEFAULTS.items():
            st.session_state[key] = value
        st.rerun()
