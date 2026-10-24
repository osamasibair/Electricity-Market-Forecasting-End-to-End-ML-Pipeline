import numpy as np
import pandas as pd
import pytest

from ensemble import best_weight, blend, weight_table, weights


def make_frame(rows=480, seed=0, lightgbm_noise=400, patchtst_noise=400):
    rng = np.random.default_rng(seed)
    actual = 30000 + 4000 * np.sin(np.arange(rows) / 7.6)
    lightgbm = actual + rng.normal(0, lightgbm_noise, rows)
    patchtst = actual + rng.normal(0, patchtst_noise, rows)
    return pd.DataFrame({
        "actual": actual,
        "lightgbm": lightgbm, "lightgbm_low": lightgbm - 600, "lightgbm_high": lightgbm + 600,
        "patchtst": patchtst, "patchtst_low": patchtst - 600, "patchtst_high": patchtst + 600,
    }, index=pd.date_range("2025-09-01", periods=rows, freq="30min"))


def test_blend_at_the_ends_is_each_model_alone():
    frame = make_frame()
    assert np.allclose(blend(frame, 0)["ensemble"], frame["lightgbm"])
    assert np.allclose(blend(frame, 1)["ensemble_high"], frame["patchtst_high"])


def test_blend_halfway_is_the_average():
    frame = make_frame()
    assert np.allclose(blend(frame, 0.5)["ensemble"], (frame["lightgbm"] + frame["patchtst"]) / 2)


def test_independent_errors_make_the_blend_beat_both_models():
    calib, test = make_frame(seed=1), make_frame(seed=2)
    table = weight_table(calib, test).set_index("weight")
    middle = table.loc[0.5, "test_mape"]
    assert middle < table.loc[0.0, "test_mape"] and middle < table.loc[1.0, "test_mape"]


def test_the_weight_is_chosen_on_the_calibration_year_only():
    calib = make_frame(seed=1, lightgbm_noise=100, patchtst_noise=900)
    test = make_frame(seed=2, lightgbm_noise=900, patchtst_noise=100)
    table = weight_table(calib, test)
    assert best_weight(table) < 0.3
    assert table.set_index("weight")["test_mape"].idxmin() > 0.7


def test_every_weight_is_recalibrated_to_about_80_percent():
    calib, test = make_frame(seed=1), make_frame(seed=2)
    table = weight_table(calib, test)
    assert len(table) == len(weights)
    assert table["test_coverage"].between(70, 90).all()
    assert table["weight"].tolist() == pytest.approx(list(weights))