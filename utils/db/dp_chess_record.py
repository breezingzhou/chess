# 东萍网象棋对局记录（ORM 版本）
from typing import Iterable, Optional
from sqlalchemy import Column, Integer, String, Text

from scripts.download_chess_record import DPChessRecord
from utils.db.common import BaseDAL, BaseModel, get_session


class DPChessRecordModel(BaseModel):
  __tablename__ = "dp_chess_record"
  id = Column(Integer, primary_key=True)
  red_player = Column(String, nullable=False)
  black_player = Column(String, nullable=False)
  type = Column(String, nullable=True)
  gametype = Column(String, nullable=True)
  result = Column(String, nullable=False)
  movelist = Column(Text, nullable=False)


class _DPChessRecordDAL(BaseDAL):
  def __init__(self):
    super().__init__(DPChessRecordModel)

  def save_record(self, record: DPChessRecord) -> None:
    if not record.chess_no:
      print("No chess_no provided; ignore")
      return
    obj = DPChessRecordModel(
        id=int(record.chess_no),
        red_player=record.red_player,
        black_player=record.black_player,
        type=record.type,
        gametype=record.gametype,
        result=record.result,
        movelist=record.movelist,
    )
    with get_session() as session:
      # use merge to perform insert or update by PK
      session.merge(obj)

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
