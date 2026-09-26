"""Clean raw data and get demand and weather onto one half hourly timeline."""

import pandas as pd
from pathlib import Path

raw = Path("data/raw")
processed = Path("data/processed")

def load_demand():
    df = pd.read_csv(raw / "demand.csv", parse_dates=["startTime"])
    df["startTime"] = df["startTime"].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df[["startTime", "initialDemandOutturn"]].rename(
        columns={"initialDemandOutturn": "demand"}
    )
    df = df.set_index("startTime").sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="30min")
    df = df.reindex(full)
    df["demand"] = df["demand"].interpolate(method="time")
    df.index.name = "timestamp"
    return df

def load_weather():
    df = pd.read_csv(raw / "weather.csv", parse_dates=["time"])
    df = df.set_index("time").sort_index()
    df = df.resample("30min").interpolate(method="time")
    df.index.name = "timestamp"
    return df

if __name__ == "__main__":
    demand = load_demand()
    weather = load_weather()
    df = demand.join(weather, how="inner")
    print(f"{len(df)} rows, {df.isna().sum().sum()} nulls")
    print(df.head())
    processed.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed / "dataset.csv")


