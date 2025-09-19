from utils.common import WORKSPACE
import sqlite3
from contextlib import contextmanager

DB_PATH = WORKSPACE / "res/chess.sqlite"


class DBManager:
  def __init__(self, db_name):
    self.db_name = db_name

  @contextmanager
  def connect(self):
    """上下文管理器，自动处理连接和事务"""
    conn = sqlite3.connect(self.db_name)
    cursor = conn.cursor()
    try:
      yield cursor
      conn.commit()
    except Exception as e:
      conn.rollback()
      raise e
    finally:
      conn.close()


DB_MANAGER = DBManager(str(DB_PATH))
