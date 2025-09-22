# %%
from _common import *
# 加载模型自我对弈 生成对弈数据
from pathlib import Path

from utils import show_images_in_slider, setup_logging, WORKSPACE
from utils.db import SelfPlayChessRecordDAL
from chess.define import ChessRecordData
from chess.game import Game
from net.policy_net import PolicyNet
from players.policy_player import PolicyPlayer

# %%
setup_logging()
# %%
checkpoint_dir = WORKSPACE / "lightning_logs/version_17/checkpoints"
checkpoint_path = list(checkpoint_dir.iterdir())[0]
# %%


def test():
  from chess.utils import generate_board_images
  model = PolicyNet.load_from_checkpoint(checkpoint_path)
  red_player = PolicyPlayer("红方", model=model, temperature=2.0)
  black_player = PolicyPlayer("黑方", model=model, temperature=2.0)
  game = Game(red_player, black_player, debug=True, evaluate=True)
  result = game.start_play_loop()
  game.log_evaluation()
  movelist = game.movelist
  images = generate_board_images(movelist, show_last_pos=True)
  show_images_in_slider(images)


# %%
def collect_selfplay_data(red_player: PolicyPlayer, black_player: PolicyPlayer, version: int, num_games: int = 1000, draw_turns: int = 200, save_epoch: int = 10):
  records: list[ChessRecordData] = []
  for i in range(num_games):
    print(f"开始第 {i + 1} / {num_games} 局对弈")
    game = Game(red_player, black_player, evaluate=True)

    winner = game.start_play_loop(draw_turns)
    r = ChessRecordData(
        red_player=red_player.display_name,
        black_player=black_player.display_name,
        winner=winner,
        movelist=game.movelist
    )
    records.append(r)
    if len(records) >= save_epoch:
      SelfPlayChessRecordDAL.save_records(records)
      records = []  # 清空已保存的记录
  SelfPlayChessRecordDAL.save_records(records)

# %%
# epoch=46-step=49585
# %%
if __name__ == "__main__":
  model = PolicyNet.load_from_checkpoint(checkpoint_path)
  red_player = PolicyPlayer("红方", model=model, temperature=2.0)
  black_player = PolicyPlayer("黑方", model=model, temperature=2.0)
  collect_selfplay_data(red_player, black_player, num_games=5000, version=1)

# %%
