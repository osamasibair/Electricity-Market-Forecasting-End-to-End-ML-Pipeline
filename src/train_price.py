"""Train a day-ahead price model and compare it with naive baselines."""
from pathlib import Path

import joblib
import lightgbm as lgb

from db import load_table
from evaluate import metrics
from tracking import log_run
from train import split_date

models = Path("models")
validation_start = "2024-09-01"
features = ["demand", "temperature_2m", "shortwave_radiation", "period", "dayofweek", "dayofyear", "is_holiday", "is_christmas", "solar_generation", "wind_onshore", "wind_offshore", "wind_total", "net_demand", "price_lag_recent", "price_lag_336", "price_roll_7d"]
candidates = {
    "all features": features,
    "no dayofyear": [f for f in features if f != "dayofyear"],
}
target = "price"


def train_price_model(train, cols):
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(train[cols], train[target])
    return model


if __name__ == "__main__":
    df = load_table("price_features")
    train = df[df.index < split_date]
    test = df[df.index >= split_date]
    print(f"train: {len(train)} rows, test: {len(test)} rows")

    fit_part = train[train.index < validation_start]
    val = train[train.index >= validation_start]
    val_mae = {}
    for name, cols in candidates.items():
        val_model = train_price_model(fit_part, cols)
        val_mae[name] = metrics(val[target], val_model.predict(val[cols]))["mae"]
        print(f"validation  {name:13} MAE {val_mae[name]:6.2f} £/MWh")
    best = min(val_mae, key=val_mae.get)
    chosen = candidates[best]
    print(f"chosen on validation: {best}")

    results = {
        "last week": metrics(test[target], test["price_lag_336"]),
        "recent day": metrics(test[target], test["price_lag_recent"]),
    }
    model = train_price_model(train, chosen)
    results["lightgbm"] = metrics(test[target], model.predict(test[chosen]))
    for name, m in results.items():
        print(f"{name:11} MAE {m['mae']:6.2f} £/MWh   RMSE {m['rmse']:6.2f} £/MWh")

    joblib.dump({"model": model, "features": chosen}, models / "price_model.joblib")
    print("saved models/price_model.joblib")

    log_run(
        experiment="price",
        params={"chosen": best, "features": ",".join(chosen), "validation_start": validation_start, "split_date": split_date},
        metrics={**{f"validation_mae_{name.replace(' ', '_')}": mae for name, mae in val_mae.items()},
                 **{f"{name.replace(' ', '_')}_{key}": value for name, m in results.items() for key, value in m.items()}},
        files=[models / "price_model.joblib"],
        model=model,
        registered_name="price-forecast",
        input_example=test[chosen].head(),
    )
