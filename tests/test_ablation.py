import pandas as pd

from ablation import merge, parse


def test_values_from_the_command_line_become_numbers_where_they_can():
    assert parse("24") == 24 and parse("0.2") == 0.2 and parse("attention") == "attention"


def test_a_new_run_replaces_only_the_rows_for_its_own_setting():
    old = pd.DataFrame({"setting": ["pooling", "pooling", "patch_len"], "value": ["flatten", "mean", "24"], "mape": [6.0, 5.8, 5.9]})
    new = pd.DataFrame({"setting": ["pooling"], "value": ["attention"], "mape": [5.5]})
    table = merge(old, new)
    assert sorted(table["value"]) == ["24", "attention"]
    assert merge(None, new).equals(new)