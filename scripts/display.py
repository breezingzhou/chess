# %%
from _common import *

# 统计各种对局指标
from chess.define import ChessWinner
from chess.utils import generate_board_images
from utils import show_images_in_slider
from utils.db import SelfPlayChessRecordDAL, SelfPlayChessRecordModel

# %%
version = 5
records = SelfPlayChessRecordDAL.query(filters=[SelfPlayChessRecordModel.version == version])
total_games = len(records)
red_win_records = [r for r in records if r.winner == ChessWinner.Red.number]
draw_win_records = [r for r in records if r.winner == ChessWinner.Draw.number]
print(f"数据版本 {version} 对局总数: {total_games}")
print(f"红方胜局数: {len(red_win_records)} 胜率: {len(red_win_records) / total_games:.2%}")
print(f"和局数: {len(draw_win_records)} 和率: {len(draw_win_records) / total_games:.2%}")
# %%
movelist = red_win_records[0].movelist
images = generate_board_images(movelist, show_last_pos=True)
show_images_in_slider(images)
