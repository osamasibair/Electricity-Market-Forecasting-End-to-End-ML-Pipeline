"""Battery backtest page, profit from scheduling a battery with each price forecast."""
import pandas as pd
import streamlit as st

from db import load_table

st.title("Battery backtest")
st.caption("A simulated 1 MW / 2 MWh battery with 90% round trip efficiency plans each days charging and discharging from a price forecast with linear programming, and the plan is scored on the real prices. Test year: 1 September 2025 to 31 August 2026.")

daily = load_table("backtest_daily")
totals = daily.sum()
summary = pd.DataFrame({
    "Profit": totals.map("£{:,.0f}".format),
    "Share of perfect foresight": (totals / totals["perfect foresight"]).map("{:.1%}".format),
    "Losing days": (daily < 0).sum(),
}).loc[totals.sort_values(ascending=False).index]
st.dataframe(summary)
st.caption("Perfect foresight plans with the real prices, so it is the most any forecast could earn.")

st.line_chart(daily.cumsum(), y_label="Cumulative profit (£)")
st.caption("Most of the profit comes from knowing the daily shape of prices: a simple average of the last 7 days captures nearly as much as the LightGBM forecast.")