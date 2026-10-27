from types import SimpleNamespace

import optuna
import pytest

from patchtst import PatchTST, PatchTSTForecaster, defaults
from tune_patchtst import best_settings, search, suggest


def test_every_suggested_setting_is_a_valid_model_setting():
    trial = optuna.trial.FixedTrial({"learning_rate": 1e-3, "weight_decay": 1e-2, "dropout": 0.1, "d_model": 32,
                                     "layers": 1, "hidden": 64, "patch_len": 48, "batch_size": 64, "lags": 1})
    settings = suggest(trial)
    assert settings["stride"] == 24
    model = PatchTSTForecaster(**settings)
    PatchTST(11, **{name: model.settings[name] for name in ["patch_len", "stride", "d_model", "layers", "hidden", "lags"]})


def test_the_first_trial_can_be_the_current_defaults():
    trial = optuna.trial.FixedTrial({name: defaults[name] for name in search})
    settings = suggest(trial)
    assert all(settings[name] == defaults[name] for name in settings)


def test_best_settings_add_the_stride_back():
    trial = SimpleNamespace(params={"patch_len": 12, "learning_rate": 1e-3})
    assert best_settings(trial) == {"patch_len": 12, "learning_rate": 1e-3, "stride": 6}


def test_a_callback_can_stop_training_early():
    from test_patchtst import make_features

    from forecast_windows import day_windows, windows

    df = make_features(days=40)
    seen = []

    def stop_after_two(epoch, loss):
        seen.append(epoch)
        if epoch == 2:
            raise optuna.TrialPruned()

    model = PatchTSTForecaster(max_epochs=5, d_model=16, hidden=32)
    with pytest.raises(optuna.TrialPruned):
        model.fit(df, windows(df.index, last="2024-02-01", stride=16), valid=day_windows(df.index, first="2024-02-01"),
                  callback=stop_after_two)
    assert seen == [1, 2]