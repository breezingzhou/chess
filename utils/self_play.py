# %%

import logging
from bz_chess import Game
from players.policy_player import PolicyPlayer
from utils.common import cal_log_epoch
from utils.db import SelfPlayChessRecord, SelfPlayChessRecordDAL


def collect_selfplay_data(red_player: PolicyPlayer, black_player: PolicyPlayer, version: int, num_games: int = 1000, draw_turns: int = 200):
  # version 表示数据版本
  logging.info(f"开始自我对弈 对弈局数：{num_games}")
  log_epoch = cal_log_epoch(num_games)
  for i in range(num_games):
    if i % log_epoch == 0:
      logging.info(f"开始第 {i + 1} / {num_games} 局对弈")
    game = Game(red_player, black_player, evaluate=False)

    winner = game.start_play_loop(draw_turns)
    r = SelfPlayChessRecord(
        id=None,
        red_player=red_player.display_name,
        black_player=black_player.display_name,
        winner=winner,
        movelist=game.movelist,
        version=version
    )
    SelfPlayChessRecordDAL.save_record(r)
