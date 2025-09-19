# 东萍网象棋对局记录
# %%
from typing import Iterable, Optional
from scripts.download_chess_record import DPChessRecord
from utils.db.common import DBManager
# %%


class _DPChessRecordDAL:
  def __init__(self, db_manager: DBManager):
    self.db_manager = db_manager

  def create_table(self) -> None:
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS dp_chess_record (
      id INTEGER PRIMARY KEY,
      red_player TEXT NOT NULL,
      black_player TEXT NOT NULL,
      type TEXT,
      gametype TEXT,
      result TEXT NOT NULL,
      movelist TEXT NOT NULL
    );
    """
    with self.db_manager.connect() as cursor:
      cursor.execute(create_table_sql)

  def save_record(self, record: DPChessRecord) -> None:
    if not record.chess_no:
      print("No chess_no provided; ignore")
      return
    sql = (
        "INSERT OR REPLACE INTO dp_chess_record"
        " (id, red_player, black_player, type, gametype, result, movelist)"
        " VALUES (?, ?, ?, ?, ?, ?, ?);"
    )
    params = (
        int(record.chess_no),
        record.red_player,
        record.black_player,
        record.type,
        record.gametype,
        record.result,
        record.movelist,
    )

    with self.db_manager.connect() as cursor:
      cursor.execute(sql, params)

  def save_records(self, records: Iterable[DPChessRecord]) -> None:
    records = [record for record in records if record.chess_no]
    insert_sql = (
        "INSERT OR REPLACE INTO dp_chess_record"
        " (id, red_player, black_player, type, gametype, result, movelist)"
        " VALUES (?, ?, ?, ?, ?, ?, ?);"
    )
    to_insert = []
    for r in records:
      to_insert.append((r.chess_no, r.red_player, r.black_player,
                        r.type, r.gametype, r.result, r.movelist))

    # Use executemany; sqlite will accept None for the primary key to auto-assign.
    with self.db_manager.connect() as cursor:
      cursor.executemany(insert_sql, to_insert)

  def get_record(self, chess_no: int) -> Optional[DPChessRecord]:
    with self.db_manager.connect() as cursor:
      cursor.execute(
          "SELECT id, red_player, black_player, type, gametype, result, movelist"
          " FROM dp_chess_record WHERE id = ?;",
          (chess_no,)
      )
      row = cursor.fetchone()
    if not row:
      return None
    id_, red_player, black_player, type_, gametype, result, movelist = row
    return DPChessRecord(
        red_player=red_player,
        black_player=black_player,
        type=type_,
        gametype=gametype,
        result=result,
        movelist=movelist,
        chess_no=id_,
    )


# %%
