# 自我对弈记录（SQLAlchemy ORM 版本）
from typing import Iterable, Optional
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped

from chess.define import ChessRecordData, ChessWinner
from utils.db.common import BaseDAL, BaseModel, get_session


@dataclass
class SelfPlayChessRecord(ChessRecordData):
  version: int
  id: Optional[int] = None
  created_at: Optional[str] = None


# TODO use Mapped to define columns
class SelfPlayChessRecordModel(BaseModel):
  __tablename__ = "selfplay_chess_record"
  id: int = Column(Integer, primary_key=True, autoincrement=True)  # type: ignore
  red_player: str = Column(String, nullable=False)  # type: ignore
  black_player: str = Column(String, nullable=False)  # type: ignore
  winner: int = Column(Integer, nullable=False)  # type: ignore
  version: int = Column(Integer, nullable=False)  # type: ignore
  movelist: str = Column(Text, nullable=False)  # type: ignore
  created_at: str = Column(TIMESTAMP, server_default=func.current_timestamp())  # type: ignore


class _SelfPlayChessRecordDAL(BaseDAL[SelfPlayChessRecordModel]):
  def __init__(self):
    super().__init__(SelfPlayChessRecordModel)

  def save_record(self, record: SelfPlayChessRecord) -> None:
    self.save_records([record])

  def save_records(self, records: Iterable[SelfPlayChessRecord]) -> None:
    objs = [
        SelfPlayChessRecordModel(
            id=r.id,
            red_player=r.red_player,
            black_player=r.black_player,
            winner=r.winner.number,
            movelist=r.movelist,
            version=r.version,
            created_at=datetime.strptime(
                r.created_at, "%Y-%m-%d %H:%M:%S") if r.created_at else None
        )
        for r in records
    ]
    with get_session() as session:
      session.add_all(objs)

  def get_record(self, id_: int) -> Optional[SelfPlayChessRecord]:
    with get_session() as session:
      obj = session.get(SelfPlayChessRecordModel, id_)
      if obj is None:
        return None

      record = SelfPlayChessRecord(
          id=obj.id,
          red_player=obj.red_player,
          black_player=obj.black_player,
          movelist=obj.movelist,
          winner=ChessWinner(obj.winner),
          version=obj.version,
          created_at=obj.created_at
      )
      return record
