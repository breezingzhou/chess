# %%
from bz_chess.board import Board
from bz_chess.define import Move
from players.base_player import BasePlayer
from pikafish import get_pikafish_engine
# %%


# %%


class PikafishPlayer(BasePlayer):
  def __init__(self, name: str, evaluate: bool = False):
    super().__init__(name, evaluate)
    self.engine = get_pikafish_engine()

  @property
  def display_name(self) -> str:
    return f"{self.name}_pikafish"

  def get_move(self, board: Board) -> Move:
    best_move_str = self.engine.bestmove(board.to_fen())
    best_move = Move.from_uci_str(best_move_str)
    return best_move
