"""Monitoring page, shows how the live demand forecasts are performing over time."""

import pandas as pd
import streamlit as st

from db import load_table


def load_optional(name):
    try:
        return load_table(name)
    except ValueError:
        return None

st.title("Demand forecast monitoring")

daily = load_table("monitoring_daily")
drift = load_table("monitoring_drift")
live = load_optional("live_forecasts")
live_daily = load_optional("monitoring_live")
latest = daily.iloc[-1] if live_daily is None else live_daily.iloc[-1]
today = pd.Timestamp.now(tz="Europe/London").date()
if live is None:
    newest, data_age = daily.index.max(), (pd.Timestamp(today) - daily.index.max()).days
else:
    newest = live.index.max().tz_localize("UTC").tz_convert("Europe/London")
    data_age = (today - newest.date()).days

left, middle, right = st.columns(3)
left.metric("7-day MAPE", f"{latest['mape_7d']:.2f}%", f"baseline {latest['baseline_mape_7d']:.2f}%", delta_color="off")
middle.metric("30-day interval coverage", f"{latest['coverage_30d']:.1f}%", "target 80%", delta_color="off")
right.metric("Latest forecast" if live is not None else "Latest data", f"{newest:%d %b %Y}", f"{max(data_age, 0)} days old", delta_color="off")

alerts = []
if latest["worse_than_baseline"]:
    alerts.append("The model has been no better than the same-time-last-week baseline over the last 7 days.")
if latest["low_coverage"]:
    alerts.append("Fewer than 70% of actual values fell inside the 80% intervals over the last 30 days.")
if live_daily is None and latest["worse_than_baseline"] and (drift.iloc[-1] > 0.25).any():
    alerts.append(f"Large drift in the latest month may explain this: {', '.join(drift.columns[drift.iloc[-1] > 0.25])}.")
if live is not None and data_age > 0:
    alerts.append(f"There has been no new forecast for {data_age} days, so the daily job may have failed.")
if live is None and data_age > 7:
    alerts.append(f"The newest data is {data_age} days old.")
for alert in alerts:
    st.warning(alert)
if not alerts:
    st.success("No alerts.")

st.subheader("Live forecasts")
if live is None:
    st.info("No live forecasts yet. The daily job adds one each morning.")
else:
    recent = live[live.index > live.index.max() - pd.Timedelta(days=7)]
    recent = recent.set_axis(recent.index.tz_localize("UTC").tz_convert("Europe/London").tz_localize(None))
    st.line_chart(recent[["actual", "forecast", "low", "high"]], y_label="Demand (MW)")
    if live_daily is None:
        st.caption("Each forecast is scored once that day's actual demand is published.")
    else:
        st.caption(f"Scored over {len(live_daily)} live days: mean MAPE {live_daily['mape'].mean():.2f}% against {live_daily['baseline_mape'].mean():.2f}% for the baseline, with {live_daily['coverage'].mean():.1f}% of actual values inside the 80% interval.")

st.header("Test year replay")

st.subheader("Daily error")
st.line_chart(daily[["mape", "baseline_mape", "mape_7d"]], y_label="MAPE (%)")
st.caption(f"Over the period, the 7-day error was no better than the baseline on {daily['worse_than_baseline'].sum()} days.")

st.subheader("Interval coverage")
st.line_chart(daily[["coverage_30d"]].assign(target=80), y_label="Coverage (%)")
st.caption(f"The 30-day coverage was below 70% on {daily['low_coverage'].sum()} days.")

st.subheader("Input drift")
st.bar_chart(drift.set_axis(drift.index.strftime("%Y-%m")), y_label="PSI vs same month in training", stack=False)
st.caption("Population stability index by month, against the same month in the training years. Below 0.1 is stable, 0.1 to 0.25 is a moderate shift, above 0.25 is a large shift. Temperature often shifts by a lot from one month to the next because weather comes in spells, so drift is used to explain changes in accuracy rather than as an alert on its own.")