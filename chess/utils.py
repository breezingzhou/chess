# %%
from pathlib import Path
from PIL import Image
from torch import Tensor

from chess.board import Board
from chess.define import ChessRecord, ChessWinner, Move, MoveTensor, StateTensor, move_to_index_tensor


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


def gen_train_data(record: ChessRecord) -> tuple[list[StateTensor], list[MoveTensor]]:
  """
  生成训练数据
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
    b.do_move(move)
  return states, move_probs


# %%
import polars as pl

res_file = Path(__file__).parent.parent / "res/大师对局.csv"


def get_chess_records() -> list[ChessRecord]:
  """获取棋谱记录"""
  if not res_file.exists():
    print(f"File not found: {res_file}")
    return []
  df = pl.read_csv(res_file, schema_overrides={"movelist": pl.String})
  records: list[ChessRecord] = []
  for row in df.iter_rows(named=True):
    try:
      record = ChessRecord(
          id=row["chess_no"],
          red_player=row["red_player"],
          black_player=row["black_player"],
          winner=ChessWinner(row["result"]),
          movelist=row["movelist"],
      )
      records.append(record)
    except Exception as e:
      print(f"Error parse chess record: {e}")
      continue
  return records


def get_policy_train_data(chess_record_num: int = 1000) -> tuple[list[StateTensor], list[MoveTensor]]:
  all_chess_records = get_chess_records()

  all_states = []
  all_move_probs = []

  chess_records = all_chess_records[:chess_record_num]
  for i, chess_record in enumerate(chess_records):
    if i % 100 == 0:
      print(f"Processing chess record {i}/{len(chess_records)}")
    try:
      states, move_probs = gen_train_data(chess_record)
      all_states.extend(states)
      all_move_probs.extend(move_probs)
    except Exception as e:
      print(f"Error generate train data in chess_no [{chess_record.id}]: {e}")
  return all_states, all_move_probs


# %%
# states, move_probs = get_policy_train_data(200)
