# 从文件或者db读取对局记录 并生成训练数据
# %%
import polars as pl
from chess import *
from chess.utils import gen_train_data
from utils.db.common import MASTER_RES_PATH

# %%


def get_chess_records() -> list[ChessRecord]:
  """获取棋谱记录"""
  if not MASTER_RES_PATH.exists():
    print(f"File not found: {MASTER_RES_PATH}")
    return []
  df = pl.read_csv(MASTER_RES_PATH, schema_overrides={"movelist": pl.String})
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
