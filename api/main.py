"""HTTP API for day ahead electricity demand forecasting."""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from evaluate import metrics
from predict import load_model, load_quantile_models, predict_day
from train import split_date

processed = Path("data/processed")

app = FastAPI(title="UK Electricity Demand Forecast")
model = load_model()
df = pd.read_csv(processed / "features.csv", index_col="timestamp", parse_dates=True)
quantile = load_quantile_models()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict", responses={404: {"description": "No data for that date"}})
def predict(day: date):
    try:
        result = predict_day(model, df, day, quantile)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    m = metrics(result["actual"], result["forecast"])
    return {
        "date": day,
        "in_training_data": day < date.fromisoformat(split_date),
        "mae": round(m["mae"]),
        "mape": round(m["mape"], 2),
        "interval": "80%",
        "forecast": [
            {"timestamp_utc": ts.isoformat(), "low": round(l), "forecast": round(f), "high": round(h), "actual": round(a)}
            for ts, f, a, l, h in zip(result.index, result["forecast"], result["actual"], result["low"], result["high"])
        ],
    }