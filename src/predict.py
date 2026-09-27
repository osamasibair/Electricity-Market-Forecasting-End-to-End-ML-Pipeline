"""Load the trained model and forecast demand for one day."""

import sys
from pathlib import Path
import joblib
import pandas as pd
from evaluate import metrics
from train import features, split_date

processed = Path("data/processed")
models = Path("models")

def load_model():
    return joblib.load(models / "model.joblib")

def predict_day(model, df, date):
    local = df.index.tz_localize("UTC").tz_convert("Europe/London")
    day = df[local.date == pd.Timestamp(date).date()]
    if day.empty:
        raise ValueError(f"no data for {date}")
    forecast = model.predict(day[features])
    return pd.DataFrame({"forecast": forecast, "actual": day["demand"]}, index=day.index)

if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-08-26"
    if pd.Timestamp(date) < pd.Timestamp(split_date):
        print("warning: this date was in the training data, so the result will look too good")
    df = pd.read_csv(processed / "features.csv", index_col="timestamp", parse_dates=True)
    result = predict_day(load_model(), df, date)
    result["error"] = result["actual"] - result["forecast"]
    print(result.round().astype(int).to_string())
    m = metrics(result["actual"], result["forecast"])
    print(f"{date}: MAE {m['mae']:.0f} MW   MAPE {m['mape']:.2f}%")