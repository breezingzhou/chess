# %%

from chess.board import Board
from chess.define import Move
from players.base_player import BasePlayer


class MCTSPlayer(BasePlayer):
  def __init__(self, name: str, c_puct: int = 5, n_playout: int = 1000):
    super().__init__(name)

  def get_action(self, board: Board) -> Move:
    pass
