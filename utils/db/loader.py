# 从文件或者db读取对局记录 并生成训练数据
# %%
from typing import Optional
import polars as pl
import logging
import torch
from torch import Tensor

from bz_chess import *
from bz_chess.utils import gen_policy_train_data, gen_value_train_data
from utils.common import cal_log_epoch
from utils.db.common import MASTER_RES_PATH
from utils.db.selfplay_chess_record import SelfPlayChessRecordModel, SelfPlayChessRecordDAL

# %%


def get_selfplay_chess_records(version: int) -> list[ChessRecord]:
  # 获取自我对弈对局记录
  # 红方是不停优化的版本
  # 黑方是固定的弱AI
  # 只获取指定版本 红方获胜的对局记录
  filters = [
      SelfPlayChessRecordModel.version == version,
      SelfPlayChessRecordModel.winner == ChessWinner.Red.number,
  ]
  model_records: list[SelfPlayChessRecordModel] = SelfPlayChessRecordDAL.query(
      filters=filters)
  records: list[ChessRecord] = []
  for m in model_records:
    records.append(m.to_chess_record())
  return records


def get_master_chess_records() -> list[ChessRecord]:
  # 获取大师对局记录
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


def get_policy_train_data(chess_record_num: Optional[int] = None, version: int = 0) -> tuple[Tensor, Tensor]:
  # 获取策略网络训练数据
  # version=0 表示从大师对局获取数据
  # version>0 表示从自我对弈获取数据
  if version == 0:
    all_chess_records = get_master_chess_records()
    all_chess_records = [r for r in all_chess_records if r.winner != ChessWinner.Draw]
    mock_opponent = False
  else:
    all_chess_records = get_selfplay_chess_records(version)
    mock_opponent = True
  if chess_record_num and chess_record_num > 0:
    chess_records = all_chess_records[:chess_record_num]
  else:
    chess_records = all_chess_records
  chess_type_str = "大师对局" if version == 0 else f"自我对弈 v{version}"
  logging.info(
      f"从[{chess_type_str}]获取[{len(all_chess_records)}]条对局记录 使用其中[{len(chess_records)}]条作为训练数据")

  all_states = []
  all_move_probs = []

  log_epoch = cal_log_epoch(len(chess_records))
  for i, chess_record in enumerate(chess_records):
    if i % log_epoch == 0:
      logging.info(f"解析对局记录 {i}/{len(chess_records)}")
    try:
      states, move_probs = gen_policy_train_data(chess_record, mock_opponent=mock_opponent)
      all_states.extend(states)
      all_move_probs.extend(move_probs)
    except Exception as e:
      logging.error(f"解析[{chess_type_str}]时候发生错误 id = {chess_record.id} error = {e}")
  return torch.stack(all_states), torch.stack(all_move_probs)

# %%


def get_value_train_chess_records(version: int) -> list[ChessRecord]:
  # 获取价值网络训练的对局记录
  assert version < 0, "特殊逻辑 version < 0 表示自我对弈生成的用于价值网络的训练数据"

  filters = [
      SelfPlayChessRecordModel.version == version,
      SelfPlayChessRecordModel.winner != ChessWinner.Draw.number,
  ]
  model_records = SelfPlayChessRecordDAL.query(
      filters=filters)
  chess_records = [m.to_chess_record() for m in model_records]

  return chess_records


def get_value_train_data(version: int, chess_record_num: Optional[int] = None, ) -> tuple[Tensor, Tensor]:
  assert version < 0, "特殊逻辑 version < 0 表示自我对弈生成的用于价值网络的训练数据"
  # 获取价值网络训练数据
  all_chess_records = get_value_train_chess_records(version)

  if chess_record_num and chess_record_num > 0:
    chess_records = all_chess_records[:chess_record_num]
  else:
    chess_records = all_chess_records

  all_states = []
  all_values = []

  log_epoch = cal_log_epoch(len(chess_records))
  for i, chess_record in enumerate(chess_records):
    if i % log_epoch == 0:
      logging.info(f"解析对局记录 {i}/{len(chess_records)}")
    try:
      states, values = gen_value_train_data(chess_record)
      all_states.extend(states)
      all_values.extend(values)
    except Exception as e:
      logging.error(f"解析[价值网络训练数据]时候发生错误 id = {chess_record.id} error = {e}")
  return torch.stack(all_states), torch.tensor(all_values)
