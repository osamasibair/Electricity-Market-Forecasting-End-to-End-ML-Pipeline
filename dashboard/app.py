"""Streamlit entry point, sets up the page and the navigation between the live and research pages."""
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

st.set_page_config(page_title="UK electricity forecasting", layout="wide")
pages = st.navigation({
    "Live": [st.Page("monitoring.py", title="Demand monitoring", default=True)],
    "Research (test year)": [
        st.Page("price_forecast.py", title="Price forecast"),
        st.Page("price_spikes.py", title="Price spikes"),
        st.Page("battery.py", title="Battery backtest"),
    ],
})
pages.run()