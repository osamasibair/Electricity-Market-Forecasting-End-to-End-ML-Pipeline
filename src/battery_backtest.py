"""Battery trading backtest, plans each days schedule from a price forecast and scores it on real prices."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linprog

processed = Path("data/processed")
models = Path("models")
split_date = "2025-09-01"
power = 1.0
capacity = 2.0
efficiency = 0.9
step = power * 0.5


def schedule_day(prices):
    n = len(prices)
    cost = np.concatenate([prices, -prices])
    cumulative = np.tril(np.ones((n, n)))
    stored = np.hstack([efficiency * cumulative, -cumulative])
    A = np.vstack([stored, -stored, np.concatenate([np.zeros(n), np.ones(n)]), np.hstack([np.eye(n), np.eye(n)])])
    b = np.concatenate([np.full(n, capacity), np.zeros(n), [capacity], np.full(n, step)])
    result = linprog(cost, A_ub=A, b_ub=b, bounds=[(0, step)] * (2 * n), method="highs")
    return result.x[:n], result.x[n:]


def profit(charge, discharge, prices):
    return float(np.sum(prices * (discharge - charge)))


def backtest(actual, forecast):
    daily = []
    for day, prices in actual.groupby(actual.index.date):
        if len(prices) != 48:
            continue
        charge, discharge = schedule_day(forecast.loc[prices.index].to_numpy())
        daily.append(profit(charge, discharge, prices.to_numpy()))
    return np.array(daily)


if __name__ == "__main__":
    df = pd.read_csv(processed / "price_features.csv", index_col=0, parse_dates=True)
    df["typical_day"] = sum(df["price"].shift(96 + 48 * k) for k in range(7)) / 7
    df["price_excess_recent"] = df["price_lag_recent"] - df["price_roll_7d"]
    test = df[df.index >= split_date].dropna()
    price_model = joblib.load(models / "price_model.joblib")
    spike_model = joblib.load(models / "spike_model.joblib")
    lightgbm = pd.Series(price_model["model"].predict(test[price_model["features"]]), index=test.index)
    spike_prob = spike_model["model"].predict_proba(test[spike_model["features"]])[:, 1]
    spike_level = test["price_roll_7d"] + spike_model["threshold"]
    forecasts = {
        "perfect foresight": test["price"],
        "LightGBM": lightgbm,
        "LightGBM + spike model": lightgbm.where(spike_prob < spike_model["cutoff"], np.maximum(lightgbm, spike_level)),
        "baseline (recent day)": test["price_lag_recent"],
        "typical day (last 7 days)": test["typical_day"],
    }
    results = {name: backtest(test["price"], forecast) for name, forecast in forecasts.items()}
    best = results["perfect foresight"].sum()
    print(f"test days: {len(results['perfect foresight'])}\n")
    for name, daily in results.items():
        print(f"{name:28} £{daily.sum():>8,.0f}   {daily.sum() / best:6.1%} of perfect   losing days: {(daily < 0).sum()}")