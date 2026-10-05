import numpy as np
import pandas as pd

from features import build_features


def make_raw(start, days): #helper, not test
    index = pd.date_range(start, periods=days * 48, freq="30min")
    return pd.DataFrame({
        "demand": np.arange(len(index), dtype=float),
        "temperature_2m": 10.0,
        "wind_speed_10m": 5.0,
        "shortwave_radiation": 0.0,
    }, index=index)

def test_no_missing_values_and_wind_dropped():
    out = build_features(make_raw("2024-01-08", 14))
    assert out.isna().sum().sum() == 0
    assert "wind_speed_10m" not in out.columns

def test_period_runs_1_to_48():
    out = build_features(make_raw("2024-01-08", 14))
    assert out["period"].min() == 1
    assert out["period"].max() == 48

def test_hdd():
    cold = make_raw("2024-01-08", 14)
    cold["temperature_2m"] = 5.0
    assert (build_features(cold)["hdd"] == 10.5).all()
    warm = make_raw("2024-01-08", 14)
    warm["temperature_2m"] = 20.0
    assert (build_features(warm)["hdd"] == 0).all()

def test_lag_recent_uses_48_before_8am_and_96_after():
    out = build_features(make_raw("2024-01-08", 14))
    gap = out["demand"] - out["demand_lag_recent"]
    early = out.index.hour < 8
    assert (gap[early] == 48).all()
    assert (gap[~early] == 96).all()

def test_christmas_flag():
    out = build_features(make_raw("2023-12-18", 16))
    assert (out.loc["2023-12-25", "is_christmas"] == 1).all()
    assert (out.loc["2023-12-31", "is_christmas"] == 1).all()
    assert (out.loc["2024-01-02", "is_christmas"] == 0).all()