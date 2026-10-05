"""Monitor the demand model day by day; accuracy against the baseline, interval coverage and drift in the inputs."""
import numpy as np
import pandas as pd

from db import load_table, save_table
from predict import load_model, load_quantile_models, predict_day
from train import split_date

drift_features = ["temperature_2m", "shortwave_radiation", "demand_lag_recent"]


def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def daily_report(results, baseline):
    day = results.index.tz_localize("UTC").tz_convert("Europe/London").date
    actual = results["actual"]
    daily = pd.DataFrame({
        "mape": ((actual - results["forecast"]).abs() / actual).groupby(day).mean() * 100,
        "baseline_mape": ((actual - baseline).abs() / actual).groupby(day).mean() * 100,
        "coverage": ((actual >= results["low"]) & (actual <= results["high"])).groupby(day).mean() * 100,
        "periods": actual.groupby(day).size(),
    })
    daily = daily[daily["periods"] >= 46].drop(columns="periods")
    daily.index = pd.to_datetime(daily.index)
    daily["mape_7d"] = daily["mape"].rolling(7, min_periods=1).mean()
    daily["baseline_mape_7d"] = daily["baseline_mape"].rolling(7, min_periods=1).mean()
    daily["coverage_30d"] = daily["coverage"].rolling(30, min_periods=1).mean()
    daily["worse_than_baseline"] = (daily["mape_7d"] >= daily["baseline_mape_7d"]).astype(int)
    daily["low_coverage"] = (daily["coverage_30d"] < 70).astype(int)
    return daily


if __name__ == "__main__":
    df = load_table("features")
    train = df[df.index < split_date]
    test = df[df.index >= split_date]
    model, quantile = load_model(), load_quantile_models()

    local_days = sorted(set(test.index.tz_localize("UTC").tz_convert("Europe/London").date))
    results = pd.concat([predict_day(model, df, day, quantile) for day in local_days])
    daily = daily_report(results, df.loc[results.index, "demand_lag_336"])

    months = test.index.to_period("M")
    drift = pd.DataFrame.from_dict(
        {month.to_timestamp(): {f: psi(train.loc[train.index.month == month.month, f], group[f]) for f in drift_features}
         for month, group in test.groupby(months) if len(group) >= 1000},
        orient="index",
    )

    save_table(daily, "monitoring_daily")
    save_table(drift, "monitoring_drift")
    print(f"monitored {len(daily)} days: mean MAPE {daily['mape'].mean():.2f}%, mean coverage {daily['coverage'].mean():.1f}%")
    print(f"days worse than baseline (7-day): {daily['worse_than_baseline'].sum()}, days with low coverage (30-day): {daily['low_coverage'].sum()}")
    print(f"highest monthly drift (PSI): {drift.max().round(3).to_dict()}")