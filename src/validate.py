"""Checking raw data quality before it reaches the model"""

import pandas as pd
from pathlib import Path

raw = Path("data/raw")

demand_min, demand_max = 10000, 60000
temp_min, temp_max = -20, 45

def check_demand(df):
    problems = []
    counts = df.groupby("settlementDate").size()
    odd = counts[~counts.isin([46, 48, 50])]
    if len(odd) > 0:
        problems.append(f"{len(odd)} dates with unexpected period counts: {odd.head().to_dict()}")
    dates = pd.to_datetime(df["settlementDate"])
    expected = pd.date_range(dates.min(), dates.max(), freq="D")
    missing = len(expected) - dates.nunique()
    if missing > 0:
        problems.append(f"{missing} dates missing entirely from the series")
    dupes = df.duplicated(subset=["settlementDate", "settlementPeriod"]).sum()
    if dupes > 0:
        problems.append(f"{dupes} duplicate settlement date/period rows")
    nulls = df["initialDemandOutturn"].isna().sum()
    if nulls > 0:
        problems.append(f"{nulls} missing demand values")
    outside = df[(df["initialDemandOutturn"] < demand_min) | (df["initialDemandOutturn"] > demand_max)]
    if len(outside) > 0:
        problems.append(f"{len(outside)} demand values outside {demand_min}-{demand_max} MW")
    return problems

def check_weather(df):
    problems = []
    times = pd.to_datetime(df["time"])
    gaps = times.diff().dropna()
    irregular = (gaps != pd.Timedelta(hours=1)).sum()
    if irregular > 0:
        problems.append(f"{irregular} gaps or overlaps in hourly sequence")
    nulls = df[["temperature_2m", "wind_speed_10m"]].isna().sum().sum()
    if nulls > 0:
        problems.append(f"{nulls} missing weather values")
    outside = df[(df["temperature_2m"] < temp_min) | (df["temperature_2m"] > temp_max)]
    if len(outside) > 0:
        n = len(outside)
        problems.append(f"{n} temperature{'s' if n != 1 else ''} outside {temp_min}-{temp_max} C")
    return problems

def check_coverage(demand, weather):
    problems = []
    d_start = pd.to_datetime(demand["startTime"]).min()
    d_end = pd.to_datetime(demand["startTime"]).max()
    w_start = pd.to_datetime(weather["time"]).min()
    w_end = pd.to_datetime(weather["time"]).max()
    if abs((d_start - w_start).days) > 1:
        problems.append(f"start dates differ: demand {d_start.date()}, weather {w_start.date()}")
    if abs((d_end - w_end).days) > 1:
        problems.append(f"end dates differ: demand {d_end.date()}, weather {w_end.date()}")
    return problems

if __name__ == "__main__":
    demand = pd.read_csv(raw / "demand.csv")
    weather = pd.read_csv(raw / "weather.csv")
    print(f"demand: {len(demand)} rows, weather: {len(weather)} rows")
    problems = check_demand(demand) + check_weather(weather)
    if problems:
        for p in problems:
            print(f"  ! {p}")
    else:
        print("  all checks passed")
        

