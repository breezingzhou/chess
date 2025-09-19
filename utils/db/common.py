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


class BaseDAL:
  def __init__(self, db_manager: DBManager):
    self.db_manager = db_manager

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if not getattr(cls, "table_name", None):
      raise TypeError(f"{cls.__name__} must define 'table_name'")

  def drop_table(self) -> None:
    with self.db_manager.connect() as cursor:
      cursor.execute(f"DROP TABLE IF EXISTS {self.__class__.table_name}")
