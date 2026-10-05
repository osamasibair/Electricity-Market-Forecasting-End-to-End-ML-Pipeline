"""Dashboard showing how the demand forecasts are performing over time."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from db import load_table

st.set_page_config(page_title="Demand forecast monitoring", layout="wide")
st.title("Demand forecast monitoring")

daily = load_table("monitoring_daily")
drift = load_table("monitoring_drift")
latest = daily.iloc[-1]
data_age = (pd.Timestamp.now().normalize() - daily.index.max()).days

left, middle, right = st.columns(3)
left.metric("7-day MAPE", f"{latest['mape_7d']:.2f}%", f"baseline {latest['baseline_mape_7d']:.2f}%", delta_color="off")
middle.metric("30-day interval coverage", f"{latest['coverage_30d']:.1f}%", "target 80%", delta_color="off")
right.metric("Latest data", f"{daily.index.max():%d %b %Y}", f"{data_age} days old", delta_color="off")

alerts = []
if latest["worse_than_baseline"]:
    alerts.append("The model has been no better than the same-time-last-week baseline over the last 7 days.")
if latest["low_coverage"]:
    alerts.append("Fewer than 70% of actual values fell inside the 80% intervals over the last 30 days.")
if latest["worse_than_baseline"] and (drift.iloc[-1] > 0.25).any():
    alerts.append(f"Large drift in the latest month may explain this: {', '.join(drift.columns[drift.iloc[-1] > 0.25])}.")
if data_age > 7:
    alerts.append(f"The newest data is {data_age} days old.")
for alert in alerts:
    st.warning(alert)
if not alerts:
    st.success("No alerts.")

st.subheader("Daily error")
st.line_chart(daily[["mape", "baseline_mape", "mape_7d"]], y_label="MAPE (%)")
st.caption(f"Over the period, the 7-day error was no better than the baseline on {daily['worse_than_baseline'].sum()} days.")

st.subheader("Interval coverage")
st.line_chart(daily[["coverage_30d"]].assign(target=80), y_label="Coverage (%)")
st.caption(f"The 30-day coverage was below 70% on {daily['low_coverage'].sum()} days.")

st.subheader("Input drift")
st.bar_chart(drift.set_axis(drift.index.strftime("%Y-%m")), y_label="PSI vs same month in training", stack=False)
st.caption("Population stability index by month, against the same month in the training years. Below 0.1 is stable, 0.1 to 0.25 is a moderate shift, above 0.25 is a large shift. Temperature often shifts by a lot from one month to the next because weather comes in spells, so drift is used to explain changes in accuracy rather than as an alert on its own.")