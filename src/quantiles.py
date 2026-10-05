"""Train quantile models that give a range of likely demand, not just one number."""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from db import load_table
from evaluate import metrics, quantile_loss
from tracking import log_run
from train import features, split_date, target

models = Path("models")
quantiles = [0.1, 0.5, 0.9]
calibration_start = "2024-09-01"


def train_quantile_models(train):
    fitted = {}
    for q in quantiles:
        model = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1)
        model.fit(train[features], train[target])
        fitted[q] = model
    return fitted


def interval_report(name, actual, low, high):
    inside = (actual >= low) & (actual <= high)
    print(f"{name:12} coverage {inside.mean() * 100:5.1f}%   "
          f"below {(actual < low).mean() * 100:4.1f}%   "
          f"above {(actual > high).mean() * 100:4.1f}%   "
          f"width {(high - low).mean():5.0f} MW")
    return inside.mean(), (high - low).mean()


if __name__ == "__main__":
    df = load_table("features")
    train = df[df.index < split_date]
    fit_part = train[train.index < calibration_start]
    calib = train[train.index >= calibration_start]
    test = df[df.index >= split_date]

    fitted = train_quantile_models(fit_part)

    calib_low = fitted[0.1].predict(calib[features])
    calib_high = fitted[0.9].predict(calib[features])
    scores = np.maximum(calib_low - calib[target], calib[target] - calib_high)
    adjustment = np.quantile(scores, 0.8)
    print(f"calibration adjustment: {adjustment:.0f} MW")

    preds = {q: model.predict(test[features]) for q, model in fitted.items()}
    for q in quantiles:
        print(f"q{int(q * 100):<3} quantile loss {quantile_loss(test[target], preds[q], q):6.0f} MW")
    print(f"median MAPE: {metrics(test[target], preds[0.5])['mape']:.2f}%")

    interval_report("raw", test[target], preds[0.1], preds[0.9])
    interval_report("calibrated", test[target], preds[0.1] - adjustment, preds[0.9] + adjustment)

    refit = train_quantile_models(train)
    refit_preds = {q: model.predict(test[features]) for q, model in refit.items()}
    print(f"refit median MAPE: {metrics(test[target], refit_preds[0.5])['mape']:.2f}%")
    coverage, width = interval_report("refit", test[target], refit_preds[0.1] - adjustment, refit_preds[0.9] + adjustment)

    joblib.dump({"models": refit, "adjustment": adjustment}, models / "quantile_models.joblib")
    print("saved models/quantile_models.joblib")

    log_run(
        experiment="demand-intervals",
        params={"quantiles": ",".join(str(q) for q in quantiles), "calibration_start": calibration_start, "split_date": split_date},
        metrics={"adjustment_mw": adjustment, "coverage": coverage, "width_mw": width,
                 "median_mape": metrics(test[target], refit_preds[0.5])["mape"]},
        files=[models / "quantile_models.joblib"],
    )