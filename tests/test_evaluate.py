import pytest
import pandas as pd
from evaluate import metrics, quantile_loss

def test_perfect_forecast_has_zero_error():
    actual = pd.Series([100.0, 200.0, 300.0])
    result = metrics(actual, actual)
    #error = actual - predicted, so 0, 0, 0
    assert result == {"mae": 0, "rmse": 0, "mape": 0}

def test_known_errors():
    actual = pd.Series([100.0, 200.0])
    predicted = pd.Series([90.0, 220.0])
    result = metrics(actual, predicted)
    assert result["mae"] == pytest.approx(15)
    assert result["rmse"] == pytest.approx(250 ** 0.5)
    assert result["mape"] == pytest.approx(10)

def test_quantile_loss_punishes_q90_underforecasts_more():
    actual = pd.Series([100.0, 100.0])
    predicted = pd.Series([90.0, 110.0])
    assert quantile_loss(actual, predicted, 0.9) == pytest.approx(5)