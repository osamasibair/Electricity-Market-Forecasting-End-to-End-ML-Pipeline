"""Build model features from the processed dataset."""

from threading import local

import pandas as pd
from pathlib import Path
import holidays

processed = Path("data/processed")
uk_holidays = holidays.UnitedKingdom(subdiv="ENG", years=range(2022, 2028))

def build_features(df):
    df = df.copy()
    local = df.index.tz_localize("UTC").tz_convert("Europe/London")
    df["period"] = local.hour * 2 + (local.minute // 30) + 1
    df["dayofweek"] = local.dayofweek
    df["dayofyear"] = local.dayofyear
    df["is_holiday"] = pd.Series(local.date, index=df.index).isin(uk_holidays).astype(int)
    df["is_christmas"] = (((local.month == 12) & (local.day >= 24)) | ((local.month == 1) & (local.day == 1))).astype(int)
    df["hdd"] = (15.5 - df["temperature_2m"]).clip(lower=0)
    df["demand_lag_recent"] = df["demand"].shift(48).where(local.hour < 8, df["demand"].shift(96))
    df["demand_lag_336"] = df["demand"].shift(336)
    df["demand_roll_96"] = df["demand"].shift(96).rolling(48).mean()
    df = df.drop(columns=["wind_speed_10m"])
    return df.dropna()

if __name__ == "__main__":
    df = pd.read_csv(processed / "dataset.csv", index_col="timestamp", parse_dates=True)
    out = build_features(df)
    pd.set_option("display.max_columns", None)
    print(f"{len(out)} rows, {out.shape[1]} columns")
    print(out.head())
    out.to_csv(processed / "features.csv")

