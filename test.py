# %%
from scripts.download_chess_record import DPChessRecord
from utils.db import DPChessRecordDAL, DPChessRecordModel
from utils.db.common import BaseModel
# %%
record = DPChessRecord(
    red_player="test_red",
    black_player="test_black",
    type="test_type",
    gametype="test_gametype",
    result="1-0",
    movelist="1. e4 e5 2. Nf3 Nc6",
    chess_no=2
)
DPChessRecordDAL.save_record(record)
# %%
res: list[DPChessRecordModel] = DPChessRecordDAL.query(
    [DPChessRecordModel.id.in_([1, 2, 3])], order_by=['-id'])
