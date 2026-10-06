"""Price spikes page, how well the classifier flags prices far above their recent level."""
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from db import load_table

st.title("Price spikes")
st.caption("A spike is a price far above its own 7-day average: the top 5% of such jumps in the training data. Results are on the test year (1 September 2025 to 31 August 2026).")

prices = load_table("price_forecasts")
prices = prices.set_axis(prices.index.tz_localize("UTC").tz_convert("Europe/London").tz_localize(None))

spike = prices["spike"]
scores = pd.DataFrame({
    "Precision": [precision_score(spike, prices["flagged"]), precision_score(spike, prices["persistence"])],
    "Recall": [recall_score(spike, prices["flagged"]), recall_score(spike, prices["persistence"])],
    "Average precision": [average_precision_score(spike, prices["spike probability"]), average_precision_score(spike, prices["persistence"])],
    "ROC AUC": [roc_auc_score(spike, prices["spike probability"]), roc_auc_score(spike, prices["persistence"])],
}, index=["LightGBM classifier", "Persistence"]).round(2)
st.dataframe(scores)
st.caption(f"{spike.mean():.1%} of test half-hours were spikes. Precision is the share of flagged half-hours that were real spikes, recall is the share of spikes that were flagged, and average precision scores the probabilities across every possible cut-off. ROC AUC is the chance a real spike gets a higher probability than a normal half-hour (0.5 is guessing), because spikes are rare, average precision is the stricter measure. Persistence flags a spike when the most recent known price was itself spike-level.")

first, last = prices.index.min().date(), prices.index.max().date()
busiest = prices["spike"].groupby(prices.index.to_period("W-SUN")).sum().idxmax().start_time.date()
start = st.date_input("Week starting", value=busiest, min_value=first, max_value=last)
week = prices.loc[pd.Timestamp(start):pd.Timestamp(start) + pd.Timedelta(days=7)]
st.line_chart(week[["actual", "spike level"]], y_label="Price (£/MWh)")
st.caption("The actual price and the spike level. A price above the spike level counts as a spike.")
st.line_chart(week[["spike probability", "cut-off", "spike"]], y_label="Spike probability")
st.caption("The classifier's spike probability for the same week. Half-hours above the cut-off are flagged, and the spike line shows when a real spike happened.")