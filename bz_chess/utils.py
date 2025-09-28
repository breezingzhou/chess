# %%
from pathlib import Path
from PIL import Image
from torch import Tensor

from bz_chess.board import Board
from bz_chess.define import ChessRecord, ChessWinner, Move, MoveTensor, StateTensor, move_to_index_tensor


# %%


def parse_movelist_str(movelist_str: str) -> list[Move]:
  move_strs = [movelist_str[i:i + 4] for i in range(0, len(movelist_str), 4)]
  moves = [Move.from_move_str(move_str) for move_str in move_strs]
  return moves


def generate_board_images(movelist_str, show_last_pos=False) -> list[Image.Image]:
  moves = parse_movelist_str(movelist_str)
  b = Board()
  images = [b.to_image()]
  for move in moves:
    b.do_move(move)
    last_pos = move.from_pos if show_last_pos else None
    img = b.to_image(last_pos=last_pos)
    images.append(img)
  return images


def gen_policy_train_data(record: ChessRecord, mock_opponent: bool = False) -> tuple[list[StateTensor], list[MoveTensor]]:
  """
  生成策略网络训练数据
  (棋盘状态编码 落子概率)
  """
  # TODO 只解析胜者或者平局的落子

  moves = parse_movelist_str(record.movelist)
  b = Board()
  states = []
  move_probs = []
  for move in moves:
    if record.winner == ChessWinner.Draw or record.winner.number == b.current_turn.number:
      # 和棋或者当前回合是胜者
      states.append(b.to_network_input())
      move_probs.append(move_to_index_tensor(move))
      if mock_opponent:
        states.append(b.to_network_input(mock_opponent=True))
        move_probs.append(move_to_index_tensor(move, mock_opponent=True))
    b.do_move(move)
  return states, move_probs


# %%
def gen_value_train_data(record: ChessRecord) -> tuple[list[StateTensor], list[float]]:
  """
  生成价值网络训练数据
  (棋盘状态编码 棋局结果)
  """
  # assert record.winner != ChessWinner.Draw, "和棋不参与价值网络训练"
  if record.winner == ChessWinner.Draw:
    return [], []
  moves = parse_movelist_str(record.movelist)
  b = Board()
  states = []
  values = []
  for i in range(len(moves) + 1):
    states.append(b.to_network_input())
    values.append(1.0 if record.winner.number == b.current_turn.number else -1.0)
    if i < len(moves):
      b.do_move(moves[i])
  return states, values

# %%
# states, move_probs = get_policy_train_data(200)
