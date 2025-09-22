# %%
from chess.define import ChessWinner
from utils.common import WORKSPACE
from utils.db import SelfPlayChessRecordDAL, SelfPlayChessRecord
import polars as pl

# %%
res_file = WORKSPACE / "res/selfplay_chess_record_20250920_001945.csv"
# %%
df = pl.read_csv(res_file, schema_overrides={"movelist": pl.String})
records: list[SelfPlayChessRecord] = []
for row in df.iter_rows(named=True):
  try:
    record = SelfPlayChessRecord(
        id=row["id"],
        red_player=row["red_player"],
        black_player=row["black_player"],
        winner=ChessWinner(row["winner"]),
        movelist=row["movelist"],
        version=1,
        created_at=row["created_at"]
    )
    records.append(record)
  except Exception as e:
    print(f"Error parse chess record: {e}")
    continue
# %%
# SelfPlayChessRecordDAL.save_records(records)
