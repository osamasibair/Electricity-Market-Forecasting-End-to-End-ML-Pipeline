"""Train a next day demand model and compare it to a baseline."""

from pathlib import Path

import joblib
import lightgbm as lgb

from db import load_table
from evaluate import metrics
from tracking import log_run

models = Path("models")
split_date = "2025-09-01"

features = ["temperature_2m", "hdd", "period", "dayofweek", "dayofyear", "is_holiday", "demand_lag_recent", "demand_lag_336", "demand_roll_96", "shortwave_radiation", "is_christmas",]
target = "demand"

def train_model(train): #LightGBM model training function
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(train[features], train[target])
    return model

if __name__ == "__main__":
    df = load_table("features")
    train = df[df.index < split_date]
    test = df[df.index >= split_date]
    print(f"train: {len(train)} rows, test: {len(test)} rows")

    baseline = metrics(test[target], test["demand_lag_336"])
    model = train_model(train)
    predictions = model.predict(test[features])
    result = metrics(test[target], predictions)

    for name, m in [("baseline", baseline), ("lightgbm", result)]:
        print(f"{name:10} MAE {m['mae']:6.0f} MW   RMSE {m['rmse']:6.0f} MW   MAPE {m['mape']:5.2f}%")

    models.mkdir(exist_ok=True)
    joblib.dump(model, models / "model.joblib")
    print("saved models/model.joblib")

    log_run(
        experiment="demand",
        params={"split_date": split_date, "features": ",".join(features), "n_estimators": model.n_estimators,
                "learning_rate": model.learning_rate, "train_rows": len(train), "test_rows": len(test)},
        metrics={f"{name}_{key}": value for name, m in [("baseline", baseline), ("lightgbm", result)] for key, value in m.items()},
        files=[models / "model.joblib"],
        model=model,
        registered_name="demand-forecast",
        input_example=test[features].head(),
    )
