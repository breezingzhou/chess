from .sqlite_to_csv import sqlite_to_csv
from .common import DB_PATH
from .dp_chess_record import _DPChessRecordDAL, DPChessRecordModel, DPChessRecord
from .selfplay_chess_record import _SelfPlayChessRecordDAL, SelfPlayChessRecord, SelfPlayChessRecordModel


SelfPlayChessRecordDAL = _SelfPlayChessRecordDAL()
DPChessRecordDAL = _DPChessRecordDAL()

SelfPlayChessRecordDAL.create_table()
DPChessRecordDAL.create_table()


__all__ = [
    "DB_PATH",
    "SelfPlayChessRecordDAL",
    "SelfPlayChessRecordModel",
    "SelfPlayChessRecord",
    "DPChessRecordDAL",
    "DPChessRecordModel",
    "DPChessRecord",
    "sqlite_to_csv",
]
