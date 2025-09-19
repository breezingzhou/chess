# %%
# 自我对弈并且通过对弈数据调整模型
# TODO 通过旋转棋盘增加数据
from _common import *
from utils.db import DPChessRecordDAL, sqlite_to_csv
# %%
sqlite_to_csv(DPChessRecordDAL.table_name)
# %%
