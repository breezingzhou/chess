# %%
from chess.board import Board
from chess.define import Move


class BasePlayer:
  def __init__(self, name: str, debug: bool = False) -> None:
    self.name = name
    self.debug = debug

  def get_move(self, board: Board) -> Move:
    raise NotImplementedError

  def log(self) -> None:
    pass
