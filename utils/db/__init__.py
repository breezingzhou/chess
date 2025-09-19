from utils.db.sqlite_to_csv import sqlite_to_csv
from utils.db.common import DB_MANAGER, DB_PATH
from utils.db.dp_chess_record import _DPChessRecordDAL
from utils.db.selfplay_chess_record import _SelfPlayChessRecordDAL, SelfPlayChessRecord

SelfPlayChessRecordDAL = _SelfPlayChessRecordDAL(DB_MANAGER)
DPChessRecordDAL = _DPChessRecordDAL(DB_MANAGER)

SelfPlayChessRecordDAL.create_table()
DPChessRecordDAL.create_table()

__all__ = [
    "DB_PATH",
    "SelfPlayChessRecordDAL",
    "DPChessRecordDAL",
    "SelfPlayChessRecord"
    "sqlite_to_csv"
]
