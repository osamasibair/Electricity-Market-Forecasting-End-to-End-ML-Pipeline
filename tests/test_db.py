import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from db import engine, load_table, save_table


def test_save_and_load_give_back_the_same_table():
    index = pd.date_range("2026-01-01", periods=3, freq="30min", name="timestamp")
    df = pd.DataFrame({"demand": [21000.5, 21500.0, 22000.25], "period": [1, 2, 3]}, index=index)
    try:
        save_table(df, "test_roundtrip")
    except OperationalError:
        pytest.skip("database isn't running")
    try:
        pd.testing.assert_frame_equal(load_table("test_roundtrip"), df, check_freq=False)
    finally:
        with engine.begin() as connection:
            connection.execute(text("DROP TABLE test_roundtrip"))