"""Build features for the day ahead price model."""

from db import load_table, save_table
from features import build_features


def build_price_features(dataset, market):
    df = build_features(dataset).join(market, how="inner")
    local = df.index.tz_localize("UTC").tz_convert("Europe/London")
    df["wind_total"] = df["wind_onshore"] + df["wind_offshore"]
    df["net_demand"] = df["demand"] - df["wind_total"] - df["solar_generation"]
    df["price_lag_recent"] = df["price"].shift(48).where(local.hour < 8, df["price"].shift(96))
    df["price_lag_336"] = df["price"].shift(336) #week baseline
    df["price_roll_7d"] = df["price"].shift(96).rolling(336).mean() #weekly rolling mean
    return df.dropna()

if __name__ == "__main__":
    dataset = load_table("dataset")
    market = load_table("market")
    out = build_price_features(dataset, market)
    print(f"{len(out)} rows, {out.shape[1]} columns")
    print(out[["price", "net_demand", "price_lag_recent", "price_lag_336", "price_roll_7d"]].head())
    save_table(out, "price_features")
