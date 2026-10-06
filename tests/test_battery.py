import numpy as np
import pandas as pd
import pytest

from battery_backtest import backtest, capacity, efficiency, profit, schedule_day, step


def test_example_day_profit():
    prices = np.array([50.0] * 16 + [80.0] * 16 + [150.0] * 8 + [70.0] * 8)
    charge, discharge = schedule_day(prices)
    assert profit(charge, discharge, prices) == pytest.approx(188.89, abs=0.01)


def test_flat_prices_no_trading():
    prices = np.full(48, 60.0)
    charge, discharge = schedule_day(prices)
    assert profit(charge, discharge, prices) == pytest.approx(0, abs=1e-6)


def test_limits_respected():
    prices = np.random.default_rng(0).normal(80, 40, 48)
    charge, discharge = schedule_day(prices)
    stored = np.cumsum(efficiency * charge - discharge)
    assert stored.max() <= capacity + 1e-6
    assert stored.min() >= -1e-6
    assert discharge.sum() <= capacity + 1e-6
    assert (charge + discharge).max() <= step + 1e-6

def test_backtest_scores_each_full_day():
    index = pd.date_range("2026-01-05", periods=96, freq="30min")
    prices = pd.Series(np.tile(np.r_[np.full(24, 50.0), np.full(24, 100.0)], 2), index=index)
    daily = backtest(prices, prices)
    assert list(daily.index) == [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")]
    assert (daily > 0).all()