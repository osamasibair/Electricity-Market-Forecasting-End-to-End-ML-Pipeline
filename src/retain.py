"""Retrain the demand models on all available data after checking a new model on the most recent weeks."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from db import load_table
from evaluate import metrics
from quantiles import train_quantile_models
from train import features, target, train_model

models = Path("models")
holdout_days = 56


if __name__ == "__main__":
    df = load_table("features")
    cutoff = df.index.max() - pd.Timedelta(days=holdout_days)
    fit_part = df[df.index <= cutoff]
    holdout = df[df.index > cutoff]
    print(f"data up to {df.index.max():%Y-%m-%d}, checking on the last {holdout_days} days ({len(holdout)} rows)")

    model = train_model(fit_part)
    new = metrics(holdout[target], model.predict(holdout[features]))
    baseline = metrics(holdout[target], holdout["demand_lag_336"])
    print(f"baseline MAPE {baseline['mape']:.2f}%   new model MAPE {new['mape']:.2f}%")
    if new["mape"] >= baseline["mape"]:
        raise SystemExit("the new model doesn't beat the baseline on recent data, so the old models are kept")

    quantile = train_quantile_models(fit_part)
    low = quantile[0.1].predict(holdout[features])
    high = quantile[0.9].predict(holdout[features])
    scores = np.maximum(low - holdout[target], holdout[target] - high)
    adjustment = float(np.quantile(scores, 0.8))
    print(f"calibration adjustment {adjustment:.0f} MW")

    models.mkdir(exist_ok=True)
    joblib.dump(train_model(df), models / "model.joblib")
    joblib.dump({"models": train_quantile_models(df), "adjustment": adjustment}, models / "quantile_models.joblib")
    report = {
        "trained_until": f"{df.index.max():%Y-%m-%d %H:%M}",
        "holdout_days": holdout_days,
        "holdout_mape": round(new["mape"], 2),
        "baseline_mape": round(baseline["mape"], 2),
        "adjustment_mw": round(adjustment),
    }
    (models / "report.json").write_text(json.dumps(report, indent=2))
    print("saved models/model.joblib, models/quantile_models.joblib and models/report.json")