# 自我对弈记录
# %%
from typing import Iterable, Optional
from dataclasses import dataclass

from chess.define import ChessRecordData, ChessWinner
from utils.db.common import BaseDAL, DBManager
# %%


@dataclass
class SelfPlayChessRecord(ChessRecordData):
  id: int
  created_at: str


class _SelfPlayChessRecordDAL(BaseDAL):
  table_name = "selfplay_chess_record"

  def __init__(self, db_manager: DBManager):
    super().__init__(db_manager)

  def create_table(self) -> None:
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS selfplay_chess_record (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      version INTEGER NOT NULL,
      red_player TEXT NOT NULL,
      black_player TEXT NOT NULL,
      winner INTEGER NOT NULL,
      movelist TEXT NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    with self.db_manager.connect() as cursor:
      cursor.execute(create_table_sql)

  def save_record(self, record: ChessRecordData) -> None:
    self.save_records([record])

  def save_records(self, records: Iterable[ChessRecordData]) -> None:
    insert_sql = (
        "INSERT INTO selfplay_chess_record"
        " (red_player, black_player, movelist, winner)"
        " VALUES (?, ?, ?, ?);"
    )
    to_insert = [(r.red_player, r.black_player, r.movelist, r.winner.number) for r in records]

    with self.db_manager.connect() as cursor:
      cursor.executemany(insert_sql, to_insert)

  def get_record(self, id_: int) -> Optional[SelfPlayChessRecord]:
    with self.db_manager.connect() as cursor:
      cursor.execute(
          "SELECT id, red_player, black_player, movelist, winner, created_at"
          " FROM selfplay_chess_record WHERE id = ?;",
          (id_,)
      )
      row = cursor.fetchone()
    if not row:
      return None
    id_, red_player, black_player, movelist, winner, created_at = row

    return SelfPlayChessRecord(
        id=id_,
        red_player=red_player,
        black_player=black_player,
        movelist=movelist,
        winner=ChessWinner(winner),
        created_at=created_at,
    )


# %%
