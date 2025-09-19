from utils.db.common import DB_MANAGER
from utils.db.dp_chess_record import _DPChessRecordDAL
from utils.db.selfplay_chess_record import _SelfPlayChessRecordDAL, SelfPlayChessRecord

SelfPlayChessRecordDAL = _SelfPlayChessRecordDAL(DB_MANAGER)
DPChessRecordDAL = _DPChessRecordDAL(DB_MANAGER)

SelfPlayChessRecordDAL.create_table()
DPChessRecordDAL.create_table()

__all__ = [
    "SelfPlayChessRecordDAL",
    "DPChessRecordDAL",
    "SelfPlayChessRecord"
]
