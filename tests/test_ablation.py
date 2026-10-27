import pandas as pd

from ablation import average, merge, parse, run_settings


def test_values_from_the_command_line_become_numbers_where_they_can():
    assert parse("24") == 24 and parse("0.2") == 0.2 and parse("attention") == "attention"


def test_a_new_run_replaces_only_the_rows_for_its_own_setting():
    old = pd.DataFrame({"setting": ["pooling", "pooling", "patch_len"], "value": ["flatten", "mean", "24"], "mape": [6.0, 5.8, 5.9]})
    new = pd.DataFrame({"setting": ["pooling"], "value": ["attention"], "mape": [5.5]})
    table = merge(old, new)
    assert sorted(table["value"]) == ["24", "attention"]
    assert merge(None, new).equals(new)


def test_patch_length_runs_keep_patches_overlapping_by_half():
    settings = run_settings({"dropout": 0.2}, "patch_len", 48, seed=1)
    assert settings == {"dropout": 0.2, "patch_len": 48, "stride": 24, "seed": 1}


def test_seeds_are_averaged_with_the_spread_of_the_test_error():
    row = average([{"mape": 5.0, "coverage": 80.0}, {"mape": 6.0, "coverage": 78.0}])
    assert row["mape"] == 5.5 and row["coverage"] == 79.0 and row["seeds"] == 2
    assert round(row["mape_std"], 3) == 0.707