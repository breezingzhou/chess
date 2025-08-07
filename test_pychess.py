# %%
from pathlib import Path
import shutil


chess_dir = Path(__file__).parent.parent / "Source Reading" / "rs-chinese-chess"
dylib_path = chess_dir / "target/debug/examples/chess_engine.dll"
target_path = Path(__file__).parent / "rs_chinese_chess" / "chess_engine.pyd"

shutil.copy(dylib_path, target_path)
print("copy chess_engine.dll to chess_engine.pyd")
# %%
from rs_chinese_chess import Board


# %%
board = Board()
moves = board.generate_move()
moves[0].pos_from.row

# %%
