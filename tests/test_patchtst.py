import numpy as np
import pandas as pd
import pytest
import torch

from evaluate import metrics
from features import build_features
from forecast_windows import (
    arrays,
    covariate_matrix,
    day_windows,
    gap_steps,
    history_steps,
    horizon,
    windows,
)
from patchtst import PatchTST, PatchTSTForecaster, RevIN, chronos_forecast

torch.set_num_threads(1)

def make_features(start="2024-01-01", days=120, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=days * 48, freq="30min")
    step = np.arange(len(index))
    temperature = 10 - 6 * np.cos(2 * np.pi * index.dayofyear / 365.25) + rng.normal(0, 2, len(index))
    daily = np.sin(2 * np.pi * (step % 48 - 14) / 48)
    demand = 28000 + 5000 * daily - 250 * temperature - 2500 * (index.dayofweek >= 5) + rng.normal(0, 300, len(index))
    raw = pd.DataFrame({"demand": demand, "temperature_2m": temperature, "wind_speed_10m": 5.0,
                        "shortwave_radiation": np.clip(daily, 0, None) * 300}, index=index)
    return build_features(raw)


def test_windows_need_a_full_week_and_the_gap_before_them():
    df = make_features(days=20)
    starts = windows(df.index)
    assert starts.min() == history_steps + gap_steps
    assert starts.max() == len(df) - horizon


def test_windows_skip_gaps_in_the_data():
    df = make_features(days=20)
    gappy = df.drop(df.index[400])
    whole = windows(df.index)
    assert len(windows(gappy.index)) < len(whole)


def test_day_windows_start_at_uk_midnight_and_skip_clock_change_days():
    df = make_features("2024-03-01", days=60)
    starts = day_windows(df.index)
    local = df.index[starts].tz_localize("UTC").tz_convert("Europe/London")
    assert ((local.hour == 0) & (local.minute == 0)).all()
    assert pd.Timestamp("2024-03-31").date() not in set(local.date)
    assert pd.Timestamp("2024-04-10").date() in set(local.date)


def test_history_ends_16_hours_before_the_forecast_day():
    df = make_features(days=20)
    start = day_windows(df.index)[:1]
    history, future, actual = arrays(df, start)
    assert history.shape == (1, history_steps) and future.shape[:2] == (1, horizon)
    assert history[0, -1] == np.float32(df["demand"].iloc[start[0] - gap_steps - 1])
    assert np.allclose(actual[0], df["demand"].iloc[start[0]:start[0] + horizon])


def test_revin_round_trip_returns_the_original_values():
    revin = RevIN()
    x = torch.randn(4, 336) * 3000 + 30000
    normalised, mean, std = revin.normalise(x)
    assert torch.allclose(normalised.mean(dim=1), torch.zeros(4), atol=1e-4)
    restored = revin.denormalise(normalised[:, :48, None], mean, std)[..., 0]
    assert torch.allclose(restored, x[:, :48], rtol=1e-4)


def test_quantiles_never_cross():
    torch.manual_seed(0)
    net = PatchTST(n_covariates=11)
    out = net(torch.randn(8, history_steps) * 3000 + 30000, torch.randn(8, horizon, 11))
    assert out.shape == (8, horizon, 3)
    assert (out[..., 0] <= out[..., 1]).all() and (out[..., 1] <= out[..., 2]).all()


def test_learns_the_daily_shape_after_a_short_training_run():
    df = make_features(days=150)
    train, test = windows(df.index, last="2024-05-01", stride=4), day_windows(df.index, first="2024-05-01")
    model = PatchTSTForecaster(max_epochs=8, d_model=32, hidden=64).fit(df, train)
    predicted = model.predict(df, test)
    actual = df["demand"].to_numpy()[test[:, None] + np.arange(horizon)]
    flat = np.full_like(actual, df["demand"].iloc[:train.max()].mean())
    assert predicted.shape == (len(test), horizon, 3)
    assert metrics(actual.ravel(), predicted[..., 1].ravel())["mape"] < metrics(actual.ravel(), flat.ravel())["mape"] / 2


def test_early_stopping_records_the_best_epoch():
    df = make_features(days=60)
    valid = day_windows(df.index, first="2024-02-20")
    model = PatchTSTForecaster(max_epochs=4, patience=2, d_model=16, hidden=32)
    model.fit(df, windows(df.index, last="2024-02-20", stride=8), valid=valid)
    assert 1 <= model.best_epochs <= len(model.valid_losses) <= 4


def test_same_seed_gives_the_same_forecast():
    df = make_features(days=30)
    train, test = windows(df.index, stride=16), day_windows(df.index)[-2:]
    first = PatchTSTForecaster(max_epochs=2, d_model=16, hidden=32).fit(df, train).predict(df, test)
    second = PatchTSTForecaster(max_epochs=2, d_model=16, hidden=32).fit(df, train).predict(df, test)
    assert np.allclose(first, second)


def test_saved_model_gives_the_same_forecast_after_loading(tmp_path):
    df = make_features(days=30)
    test = day_windows(df.index)[-2:]
    model = PatchTSTForecaster(max_epochs=1, d_model=16, hidden=32).fit(df, windows(df.index, stride=16))
    model.save(tmp_path / "patchtst.pt", adjustment=123.0)
    loaded, saved = PatchTSTForecaster.load(tmp_path / "patchtst.pt")
    assert saved["adjustment"] == 123.0
    assert np.allclose(model.predict(df, test), loaded.predict(df, test))


def test_unknown_settings_are_rejected():
    with pytest.raises(ValueError):
        PatchTSTForecaster(layer=3)


def test_covariates_include_sin_and_cos_of_the_calendar():
    df = make_features(days=10)
    assert covariate_matrix(df).shape == (len(df), 5 + 2 * 3)


def test_chronos_forecast_uses_only_history_before_the_cutoff_and_keeps_the_forecast_day():
    class FakePipeline:
        def predict_quantiles(self, inputs, prediction_length, quantile_levels):
            self.last_values = [float(x[-1]) for x in inputs]
            values = torch.arange(prediction_length, dtype=torch.float32)
            return values[None, :, None].repeat(len(inputs), 1, len(quantile_levels)), None

    df = make_features(days=20)
    starts = day_windows(df.index)[:3]
    pipeline = FakePipeline()
    out = chronos_forecast(df, starts, pipeline, context=500)
    assert out.shape == (3, horizon, 3)
    assert out[0, 0, 0] == gap_steps
    assert pipeline.last_values[-1] == pytest.approx(df["demand"].iloc[starts[-1] - gap_steps - 1])

def test_forecast_table_keeps_every_model_next_to_the_actual_demand():
    from train_patchtst import forecast_table
    index = pd.date_range("2025-09-01", periods=4, freq="30min")
    table = forecast_table(index, np.array([1.0, 2, 3, 4]), {"lightgbm": np.zeros(4), "patchtst": np.ones(4)})
    assert list(table.columns) == ["actual", "lightgbm", "patchtst"]
    assert table.index.equals(index) and table["actual"].iloc[-1] == 4