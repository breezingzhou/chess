# %%
# 加载模型自我对弈 生成对弈数据
from torch import Tensor
from chess.board import Board

from chess.define import MOVE_TO_INDEX
from chess.game import Game
from net.policy_net import PolicyNet
from players.policy_player import PolicyPlayer
from pathlib import Path

# %%
from chess.utils import generate_board_images
from utils import show_images_in_slider, setup_logging
# %%
setup_logging()
# %%
checkpoint_dir = Path("lightning_logs/version_17/checkpoints")
checkpoint_path = list(checkpoint_dir.iterdir())[0]
# %%


def test():
  model = PolicyNet.load_from_checkpoint(checkpoint_path)
  red_player = PolicyPlayer("红方", model=model, temperature=2.0)
  black_player = PolicyPlayer("黑方", model=model, temperature=2.0)
  game = Game(red_player, black_player, debug=True)
  result = game.start_play_loop()
  game.log()
  movelist = game.movelist
  images = generate_board_images(movelist, show_last_pos=True)
  show_images_in_slider(images)

# %%

# %%
