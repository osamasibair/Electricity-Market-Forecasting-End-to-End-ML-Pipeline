"""Read and write pipeline tables in postgresql."""
import os

import pandas as pd
from sqlalchemy import create_engine

database_url = os.environ.get("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/electricity")
engine = create_engine(database_url, pool_pre_ping=True)


def save_table(df, name):
    df.to_sql(name, engine, if_exists="replace", index_label="timestamp", chunksize=10_000)


def load_table(name):
    return pd.read_sql_table(name, engine, index_col="timestamp")