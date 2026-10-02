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

def load_prices():
    df = pd.read_csv(raw / "prices.csv", parse_dates=["startTime"])
    df["startTime"] = df["startTime"].dt.tz_convert("UTC").dt.tz_localize(None)
    df.loc[df["volume"] == 0, "price"] = None
    df = df[["startTime", "price"]].set_index("startTime").sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="30min")
    df = df.reindex(full)
    df["price"] = df["price"].interpolate(method="time")
    df.index.name = "timestamp"
    return df

def load_generation():
    df = pd.read_csv(raw / "generation.csv", parse_dates=["startTime", "publishTime"])
    df = df.sort_values("publishTime").drop_duplicates(["startTime", "psrType"], keep="last")
    df["startTime"] = df["startTime"].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.pivot(index="startTime", columns="psrType", values="quantity")
    df = df.rename(columns={"Wind Onshore": "wind_onshore", "Wind Offshore": "wind_offshore", "Solar": "solar_generation"})
    df.columns.name = None
    df = df.clip(lower=0)
    full = pd.date_range(df.index.min(), df.index.max(), freq="30min")
    df = df.reindex(full).interpolate(method="time")
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

    market = load_prices().join(load_generation(), how="inner")
    print(f"market: {len(market)} rows, {market.isna().sum().sum()} nulls")
    print(market.head())
    market.to_csv(processed / "market.csv")

