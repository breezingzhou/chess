# 东萍网象棋对局记录（ORM 版本）
from dataclasses import dataclass
from typing import Iterable, Optional, cast
from sqlalchemy import Column, Integer, String, Text

from utils.db.common import BaseDAL, BaseModel, get_session

# 东萍象棋网 对局记录


@dataclass
class DPChessRecord:
  red_player: str
  black_player: str
  type: str | None
  gametype: str | None
  result: str
  movelist: str
  chess_no: int = 0


class DPChessRecordModel(BaseModel):
  __tablename__ = "dp_chess_record"
  id: int = Column(Integer, primary_key=True)  # type: ignore
  red_player: str = Column(String, nullable=False)  # type: ignore
  black_player: str = Column(String, nullable=False)  # type: ignore
  type: Optional[str] = Column(String, nullable=True)  # type: ignore
  gametype: Optional[str] = Column(String, nullable=True)  # type: ignore
  result: str = Column(String, nullable=False)  # type: ignore
  movelist: str = Column(Text, nullable=False)  # type: ignore


class _DPChessRecordDAL(BaseDAL[DPChessRecordModel]):
  def __init__(self):
    super().__init__(DPChessRecordModel)

  def save_record(self, record: DPChessRecord) -> None:
    self.save_records([record])

  def save_records(self, records: Iterable[DPChessRecord]) -> None:
    objs = []
    for r in records:
      if not r.chess_no:
        continue
      objs.append(DPChessRecordModel(
          id=int(r.chess_no),
          red_player=r.red_player,
          black_player=r.black_player,
          type=r.type,
          gametype=r.gametype,
          result=r.result,
          movelist=r.movelist,
      ))
    with get_session() as session:
      for o in objs:
        # insert or update by PK
        session.merge(o)

  def get_record(self, chess_no: int) -> Optional[DPChessRecord]:
    with get_session() as session:
      obj = session.get(DPChessRecordModel, chess_no)
      if obj is None:
        return None
      return DPChessRecord(
          red_player=obj.red_player,
          black_player=obj.black_player,
          type=obj.type,
          gametype=obj.gametype,
          result=obj.result,
          movelist=obj.movelist,
          chess_no=obj.id,
      )
