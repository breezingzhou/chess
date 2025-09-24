# %%
from typing import cast
from _common import *
from chess.define import ChessWinner
from utils.common import WORKSPACE
from utils.db import SelfPlayChessRecordDAL, SelfPlayChessRecord, SelfPlayChessRecordModel
import polars as pl
# %%
res_file = WORKSPACE / "res/selfplay_chess_record_20250920_001945.csv"
# %%


def load_records_from_csv(res_file: Path) -> list[SelfPlayChessRecord]:
  df = pl.read_csv(res_file, schema_overrides={"movelist": pl.String})
  records: list[SelfPlayChessRecord] = []
  for row in df.iter_rows(named=True):
    try:
      record = SelfPlayChessRecord(
          id=row["id"],
          red_player=row["red_player"],
          black_player=row["black_player"],
          winner=ChessWinner(row["winner"]),
          movelist=row["movelist"],
          version=1,
          created_at=row["created_at"]
      )
      records.append(record)
    except Exception as e:
      print(f"Error parse chess record: {e}")
      continue
  return records


# %%
def test_mcts():
  from chess.board import Board
  from net.policy_net import PolicyNet
  from net.value_net import ValueNet
  from utils.common import PolicyCheckPointDir, ValueCheckPointDir
  from utils.mcts import run_mcts

  b = Board()
  pnet = PolicyNet.load_from_checkpoint(PolicyCheckPointDir / "version_5.ckpt")
  vnet = ValueNet.load_from_checkpoint(ValueCheckPointDir / "version_1.ckpt")
  move, pi, root = run_mcts(b, pnet, vnet, n_simulations=50, temperature=1.0)
  print("Chosen move:", move, "sum(pi)=", float(pi.sum()))
  # 打印前若干子节点统计
  stats = sorted([(move, node.visits, round(node.q_value(), 3), round(node.prior, 3))
                 for move, node in root.child_stats()], key=lambda x: -x[1])[:5]
  for s in stats:
    print(f"move={s[0]}", f"N={s[1]}", f"Q={s[2]}", f"P={s[3]}")
# %%


test_mcts()
# %%
