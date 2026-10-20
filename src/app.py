"""Router: page config, shared state and sidebar, then the selected page. Pages live in src/pages/."""

import streamlit as st

from stockapp.advice import BANNER
from stockapp_ui import state
from stockapp_ui.router import available_pages

st.set_page_config(page_title="Fractional Kelly Analyzer", layout="wide")
state.init_state()

pages = [
    st.Page(spec.path, title=spec.title, icon=spec.icon, url_path=spec.url_path, default=i == 0)
    for i, spec in enumerate(available_pages())
]
navigation = st.navigation(pages)
state.render_sidebar()

st.title("📈 Fractional Kelly Stock Analyzer")
st.warning(BANNER)
navigation.run()
