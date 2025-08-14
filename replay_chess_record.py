# 棋谱回放
# 生成训练数据
# %%

from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image

from define import N_FEATURES, Position, Chess, ChessType, ChessColor, Action, Move, BOARD_WIDTH, BOARD_HEIGHT
from board import Board
# %%
save_file = Path(__file__).parent / "res/大师对局.dat"


def get_chess_records() -> list[tuple[int, str]]:
  """获取棋谱记录"""
  if not save_file.exists():
    return []
  with open(save_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
  records = []
  for index, line in enumerate(lines):
    parts = line.strip().split(" ")
    if len(parts) == 2:
      chess_no = int(parts[0])
      movelist = parts[1]
      records.append((chess_no, movelist))
    else:
      print(f"Invalid record line at {index}: {line.strip()}")
      continue
  return records


# %%


def generate_board_images(movelist_str) -> list[Image.Image]:
  move_strs = [movelist_str[i:i + 4] for i in range(0, len(movelist_str), 4)]
  moves = [Move.from_str(move_str) for move_str in move_strs]
  b = Board()
  images = [b.to_image()]
  for move in moves:
    b.do_move(move)
    img = b.to_image()
    images.append(img)
  return images


# %%
from utils import show_images_in_slider

chess_records = get_chess_records()
movelist_str = chess_records[0][1]
images = generate_board_images(movelist_str)
show_images_in_slider(images)

# %%
