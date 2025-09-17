# %%
from chess.board import Board
from chess.define import Move


class BasePlayer:
  def __init__(self, name: str):
    self.name = name

  def get_move(self, board: Board) -> Move:
    raise NotImplementedError
