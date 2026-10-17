import numpy as np
import pandas as pd
import pytest

from monitor import daily_report, psi, live_adjustment


def test_psi_is_small_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    assert psi(reference, rng.normal(0, 1, 5000)) < 0.1
    assert psi(reference, rng.normal(2, 1, 5000)) > 0.25


def test_daily_report_scores_a_day():
    index = pd.date_range("2026-01-05", periods=48, freq="30min")
    results = pd.DataFrame({"actual": 100.0, "forecast": 90.0, "low": 95.0, "high": 105.0}, index=index)
    daily = daily_report(results, pd.Series(80.0, index=index))
    assert len(daily) == 1
    assert daily["mape"].iloc[0] == pytest.approx(10)
    assert daily["baseline_mape"].iloc[0] == pytest.approx(20)
    assert daily["coverage"].iloc[0] == 100
    assert daily["worse_than_baseline"].iloc[0] == 0



def test_live_adjustment_widens_intervals_until_80_percent_are_covered():  
    index = pd.date_range("2026-09-01", periods=48 * 30, freq="30min")  
    misses = np.tile(np.arange(10) * 100.0, len(index) // 10)  
    live = pd.DataFrame({"low": 900.0, "high": 1100.0, "actual": 1100.0 + misses, "extra": 0.0}, index=index) 
    assert live_adjustment(live) == pytest.approx(720)  


def test_live_adjustment_ignores_its_own_earlier_widening():  
    index = pd.date_range("2026-09-01", periods=48 * 30, freq="30min")  
    live = pd.DataFrame({"low": 800.0, "high": 1200.0, "actual": 1150.0, "extra": 100.0}, index=index)  
    assert live_adjustment(live) == pytest.approx(50)  


def test_live_adjustment_waits_for_a_week_of_scored_forecasts():  
    index = pd.date_range("2026-09-01", periods=48 * 3, freq="30min")  
    live = pd.DataFrame({"low": 900.0, "high": 1100.0, "actual": 2000.0}, index=index)  
    assert live_adjustment(live) == 0.0  