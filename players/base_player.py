# %%
from chess.board import Board
from chess.define import Move


class BasePlayer:
  def __init__(self, name: str, evaluate: bool = False) -> None:
    self.name = name
    # 是否是评估模式 评估模式下记录topk等信息
    self.evaluate = evaluate

  @property
  def display_name(self) -> str:
    return f"{self.name}"

  def get_move(self, board: Board) -> Move:
    raise NotImplementedError

  def log_evaluation(self) -> None:
    pass
