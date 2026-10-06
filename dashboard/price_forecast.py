"""Price forecast page, the LightGBM price forecast against the actual price and two baselines."""
import pandas as pd
import streamlit as st

from db import load_table

st.title("Price forecast")
st.caption("Half hourly Market Index Price over the test year (1 September 2025 to 31 August 2026). This model isn't run live, because live price forecasts would need wind and solar generation forecasts.")

prices = load_table("price_forecasts")
prices = prices.set_axis(prices.index.tz_localize("UTC").tz_convert("Europe/London").tz_localize(None))

errors = prices[["LightGBM", "recent day"]].sub(prices["actual"], axis=0)
scores = pd.DataFrame({
    "MAE (£/MWh)": errors.abs().mean(),
    "RMSE (£/MWh)": (errors ** 2).mean() ** 0.5,
}).round(2)
st.dataframe(scores)
st.caption("Lower is better. Prices can be negative, so the forecasts are scored in £/MWh rather than percentage error.")

first, last = prices.index.min().date(), prices.index.max().date()
start = st.date_input("Week starting", value=pd.Timestamp("2026-01-12").date(), min_value=first, max_value=last)
week = prices.loc[pd.Timestamp(start):pd.Timestamp(start) + pd.Timedelta(days=7)]
st.line_chart(week[["actual", "LightGBM", "recent day"]], y_label="Price (£/MWh)")
st.caption("The actual price against the LightGBM forecast and the recent day baseline (the most recent known price at the same time of day). Pick any week in the test year.")