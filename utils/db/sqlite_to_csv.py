# %%
import polars as pl
import sqlite3
from datetime import datetime
from .common import DB_PATH
# %%


def sqlite_to_csv(table_name: str):
  conn = sqlite3.connect(DB_PATH)
  query = f"SELECT * FROM {table_name};"
  df = pl.read_database(query, conn)
  csv_file = DB_PATH.parent / f"{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
  df.write_csv(csv_file)


# %%
