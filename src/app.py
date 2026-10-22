"""Router: page config, shared state and sidebar, then the selected page. Page scripts live in src/app_pages/."""

import streamlit as st

from stockapp.advice import BANNER
from stockapp.logging_config import configure_logging
from stockapp_ui import auth, diagnostics, state
from stockapp_ui.router import available_pages

configure_logging()
diagnostics.install()
st.set_page_config(page_title="Fractional Kelly Analyzer", layout="wide")
state.init_state()
identity = auth.current_identity()
diagnostics.set_session_user(identity.sub_hash if identity else None)

pages = [
    st.Page(spec.path, title=spec.title, icon=spec.icon, url_path=spec.url_path, default=i == 0)
    for i, spec in enumerate(available_pages())
]
navigation = st.navigation(pages)
state.render_sidebar()
auth.render_sidebar_controls()

st.title("📈 Fractional Kelly Stock Analyzer")
st.warning(BANNER)
navigation.run()
