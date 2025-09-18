# %%
import sqlite3
from typing import Iterable, Optional

from scripts.download_chess_record import DPChessRecord
from pathlib import Path

# %%
DB_PATH = Path(__file__).parent.parent / "res/chess.sqlite"
# %%
CREATE_TABLE_SQL = """
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


def init_db(db_path: str = str(DB_PATH)) -> sqlite3.Connection:
  """Initialize and return a sqlite3 connection. Ensures foreign keys enabled."""
  conn = sqlite3.connect(db_path)
  conn.execute("PRAGMA foreign_keys = ON;")
  return conn


def create_table(conn: sqlite3.Connection) -> None:
  """Create the `dp_chess_record` table if it doesn't exist."""
  conn.executescript(CREATE_TABLE_SQL)
  conn.commit()


def save_record(conn: sqlite3.Connection, record: DPChessRecord) -> None:
  """Save a single DPChessRecord into the database.

  Maps `DPChessRecord.chess_no` to table `id`. If `chess_no` is falsy, an INSERT without `id` is used
  so sqlite will auto-assign a primary key. Otherwise we use INSERT OR REPLACE on `id`.
  """
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

  cur = conn.cursor()
  cur.execute(sql, params)
  conn.commit()


def save_records(conn: sqlite3.Connection, records: Iterable[DPChessRecord]) -> None:
  """Save multiple records in a single transaction for efficiency."""
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
  cur = conn.cursor()
  cur.executemany(insert_sql, to_insert)
  conn.commit()


def get_record(conn: sqlite3.Connection, chess_no: int) -> Optional[DPChessRecord]:
  """Fetch a record by id (mapped from chess_no) and return DPChessRecord or None."""
  cur = conn.cursor()
  cur.execute(
      "SELECT id, red_player, black_player, type, gametype, result, movelist"
      " FROM dp_chess_record WHERE id = ?;",
      (chess_no,)
  )
  row = cur.fetchone()
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
