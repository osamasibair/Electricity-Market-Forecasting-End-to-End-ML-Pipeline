"""Forecast tomorrows demand from the latest data, and score earlier forecasts once the actual demand is known."""
import pandas as pd

from db import load_table, save_table
from features import build_features
from ingest import fetch_demand, fetch_weather_forecast
from monitor import daily_report
from predict import load_model, load_quantile_models, predict_day

history_days = 21


def latest_demand(start, end):
    raw = fetch_demand(start, end)
    raw["startTime"] = pd.to_datetime(raw["startTime"]).dt.tz_convert("UTC").dt.tz_localize(None)
    demand = raw.drop_duplicates("startTime", keep="last").set_index("startTime")["initialDemandOutturn"].sort_index()
    full = pd.date_range(demand.index.min(), demand.index.max(), freq="30min")
    return demand.reindex(full).interpolate(method="time").rename("demand")


def latest_weather():
    raw = fetch_weather_forecast(past_days=history_days, forecast_days=3)
    raw["time"] = pd.to_datetime(raw["time"])
    return raw.set_index("time").sort_index().resample("30min").interpolate(method="time")


if __name__ == "__main__":
    today = pd.Timestamp.now(tz="Europe/London").normalize()
    tomorrow = (today + pd.Timedelta(days=1)).date()
    demand = latest_demand(f"{today - pd.Timedelta(days=history_days):%Y-%m-%d}", f"{today:%Y-%m-%d}")
    df = build_features(latest_weather().join(demand))

    forecast = predict_day(load_model(), df, tomorrow, load_quantile_models())
    if len(forecast) < 46:
        raise SystemExit(f"only {len(forecast)} half-hours of {tomorrow} could be forecast, so some recent data is missing")
    forecast["baseline"] = df.loc[forecast.index, "demand_lag_336"]

    try:
        earlier = load_table("live_forecasts")
        live = pd.concat([earlier[~earlier.index.isin(forecast.index)], forecast]).sort_index()
    except ValueError:
        live = forecast
    live["actual"] = demand.reindex(live.index).fillna(live["actual"])
    save_table(live, "live_forecasts")
    print(f"forecast {tomorrow}: {forecast['forecast'].min():.0f} to {forecast['forecast'].max():.0f} MW")

    scored = live.dropna(subset=["actual"])
    daily = daily_report(scored, scored["baseline"])
    if len(daily):
        save_table(daily, "monitoring_live")
        print(f"scored {len(daily)} live days: mean MAPE {daily['mape'].mean():.2f}% (baseline {daily['baseline_mape'].mean():.2f}%), mean coverage {daily['coverage'].mean():.1f}%")
    else:
        print("no complete days to score yet")