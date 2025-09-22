# %%
# 自我对弈并且通过对弈数据调整模型
# TODO 训练的logs及模型自定义保存位置
from _common import *
from utils.db import DPChessRecordDAL, sqlite_to_csv
# %%
# sqlite_to_csv(DPChessRecordDAL.model.__tablename__)
# %%
from chess import Board, Game, Move, Position, Chess, ChessColor, BOARD_WIDTH, BOARD_HEIGHT
from chess.board import test_mock_opponent

test_mock_opponent()
# %%
