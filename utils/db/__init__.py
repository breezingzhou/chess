from .sqlite_to_csv import sqlite_to_csv
from .common import DB_PATH, MASTER_RES_PATH
from .dp_chess_record import DPChessRecordDAL, DPChessRecordModel, DPChessRecord
from .selfplay_chess_record import SelfPlayChessRecord, SelfPlayChessRecordDAL, SelfPlayChessRecordModel
from .loader import get_selfplay_chess_records, get_master_chess_records, get_policy_train_data, \
    get_value_train_chess_records, get_value_train_data


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
    "get_selfplay_chess_records",
    "get_master_chess_records",
    "get_policy_train_data",
    "get_value_train_chess_records",
    "get_value_train_data",
]
