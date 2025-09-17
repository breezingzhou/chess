# %%
from pathlib import Path
from PIL import Image
from torch import Tensor

from .board import Board
from .define import Move, MoveTensor, StateTensor, move_to_index_tensor

# %%

# %%


def parse_movelist_str(movelist_str: str) -> list[Move]:
  move_strs = [movelist_str[i:i + 4] for i in range(0, len(movelist_str), 4)]
  moves = [Move.from_move_str(move_str) for move_str in move_strs]
  return moves


def generate_board_images(movelist_str) -> list[Image.Image]:
  moves = parse_movelist_str(movelist_str)
  b = Board()
  images = [b.to_image()]
  for move in moves:
    b.do_move(move)
    img = b.to_image()
    images.append(img)
  return images


def gen_train_data(movelist_str: str) -> tuple[list[StateTensor], list[MoveTensor]]:
  """
  生成训练数据
  (棋盘状态编码 落子概率)
  """
  # TODO 只解析胜者或者平局的落子

  moves = parse_movelist_str(movelist_str)
  b = Board()
  states = []
  move_probs = []
  for move in moves:
    states.append(b.to_network_input())
    move_probs.append(move_to_index_tensor(move))
    b.do_move(move)
  return states, move_probs

# %%


save_file = Path(__file__).parent.parent / "res/大师对局.dat"


def get_chess_records() -> dict[int, str]:
  """获取棋谱记录"""
  if not save_file.exists():
    return {}
  with open(save_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
  records = {}
  for index, line in enumerate(lines):
    parts = line.strip().split(" ")
    if len(parts) == 2:
      chess_no = int(parts[0])
      movelist = parts[1]
      records[chess_no] = movelist
    else:
      print(f"Invalid record line at {index}: {line.strip()}")
      continue
  return records


def get_policy_train_data() -> tuple[list[StateTensor], list[MoveTensor]]:
  chess_records = get_chess_records()
  all_states = []
  all_move_probs = []
  for chess_no, movelist_str in chess_records.items():
    try:
      states, move_probs = gen_train_data(movelist_str)
      all_states.extend(states)
      all_move_probs.extend(move_probs)
    except Exception as e:
      print(f"Error generate train data in chess_no [{chess_no}]: {e}")
  return all_states, all_move_probs
