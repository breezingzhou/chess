# %%
from utils.db import DPChessRecordDAL
# %%
res = DPChessRecordDAL.query_by_id(1)
res
# %%
DPChessRecordDAL.create_table()
