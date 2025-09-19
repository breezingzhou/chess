# 自我对弈记录（SQLAlchemy ORM 版本）
from typing import Iterable, Optional
from dataclasses import dataclass

from sqlalchemy import Column, Integer, String, Text, TIMESTAMP
from sqlalchemy.sql import func

from chess.define import ChessRecordData, ChessWinner
from utils.db.common import BaseDAL, BaseModel, get_session


@dataclass
class SelfPlayChessRecord(ChessRecordData):
  id: int
  created_at: str


class SelfPlayChessRecordModel(BaseModel):
  __tablename__ = "selfplay_chess_record"
  id = Column(Integer, primary_key=True, autoincrement=True)
  version = Column(Integer, nullable=False, default=0)
  red_player = Column(String, nullable=False)
  black_player = Column(String, nullable=False)
  winner = Column(Integer, nullable=False)
  movelist = Column(Text, nullable=False)
  created_at = Column(TIMESTAMP, server_default=func.current_timestamp())


class _SelfPlayChessRecordDAL(BaseDAL):
  def __init__(self):
    super().__init__(SelfPlayChessRecordModel)

  def save_record(self, record: ChessRecordData) -> None:
    self.save_records([record])

  def save_records(self, records: Iterable[ChessRecordData]) -> None:
    objs = [
        SelfPlayChessRecordModel(
            red_player=r.red_player,
            black_player=r.black_player,
            movelist=r.movelist,
            winner=r.winner.number if hasattr(r.winner, 'number') else int(r.winner)
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
      return SelfPlayChessRecord(
          id=obj.id,
          red_player=obj.red_player,
          black_player=obj.black_player,
          movelist=obj.movelist,
          winner=ChessWinner(obj.winner),
          created_at=str(obj.created_at),
      )
