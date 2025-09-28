# %%
from _common import *
from bz_chess import ChessWinner, Game
from bz_chess.board import Board
from bz_chess.define import MOVE_TO_INDEX
from bz_chess.utils import gen_value_train_data, generate_board_images
from players.pikafish_player import PikafishPlayer
from players.policy_player import PolicyPlayer
from utils.common import WORKSPACE
from utils.db import SelfPlayChessRecordDAL, SelfPlayChessRecord, SelfPlayChessRecordModel
import polars as pl

from net.policy_net import PolicyNet
from net.value_net import ValueNet
from utils.common import PolicyCheckPointDir, ValueCheckPointDir
from utils.mcts import MCTS
from utils.mcts_go import MCTSGo
from collections import defaultdict

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
  from bz_chess.board import Board
  from net.policy_net import PolicyNet
  from net.value_net import ValueNet
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
def test_gen_value_train_data():
  filters = [
      SelfPlayChessRecordModel.version == -1,
      SelfPlayChessRecordModel.winner != ChessWinner.Draw.number,
  ]
  model_records = SelfPlayChessRecordDAL.query(
      filters=filters)
  record = model_records[0].to_chess_record()
  # records = [r.to_chess_record() for r in model_records]
  print(f"winner:{record.winner}  movelist:{record.movelist}")
  print(f"steps count: {len(record.movelist) / 4}")
  states, values = gen_value_train_data(record)
  print(f" {len(states)} {len(values)}")


# %%
def test_mctsplayer():
  from players.mstc_player import MCTSPlayer
  from utils import show_images_in_slider

  policy_net = PolicyNet.load_from_checkpoint(PolicyCheckPointDir / "version_5.ckpt")
  value_net = ValueNet.load_from_checkpoint(ValueCheckPointDir / "version_1.ckpt")

  red_player = MCTSPlayer("Red", policy_net, value_net, temperature=1.0)
  black_player = MCTSPlayer("Black", policy_net, value_net, temperature=1.0)
  game = Game(red_player, black_player, evaluate=False, debug=True)
  winner = game.start_play_loop(draw_turns=200)
  movelist = game.movelist
  show_images_in_slider(generate_board_images(movelist, show_last_pos=True))


# %%
from utils.mcts import MCTSNode


def test_mcts_root():
  import torch
  import torch.nn.functional as F
  policy_net = PolicyNet.load_from_checkpoint(PolicyCheckPointDir / "version_5.ckpt")
  value_net = ValueNet.load_from_checkpoint(ValueCheckPointDir / "version_1.ckpt")
  board = Board()
  mcts = MCTS(policy_net, value_net)
  root = mcts.search(board, n_simulations=1000, add_dirichlet=False)
  # policy = mcts.get_policy(root, temperature=self.temperature)


def show_root(root: MCTSNode):
  current = [root]
  next = []
  level = 0
  levels = defaultdict(int)
  while current:
    level += 1
    for node in current:
      if node.children:
        levels[level] += 1
      next.extend(node.children.values())
    current, next = next, []
  print(levels)


# %%
policy_net = PolicyNet.load_from_checkpoint(PolicyCheckPointDir / "version_5.ckpt")
red_player = PolicyPlayer("Red", policy_net)
black_player = PikafishPlayer("Black")

game = Game(red_player, black_player, evaluate=True, debug=True)
game.board.to_image().show()
move = game._play_one_turn()
print(f"move: {move}")
game.board.to_image().show()
move = game._play_one_turn()
print(f"move: {move}")
game.board.to_image().show()
# %%
