# %%
from copy import deepcopy
from itertools import product
from torch import Tensor

from define import Position, Chess, ChessType, ChessColor, Action, Move, BOARD_WIDTH, BOARD_HEIGHT
# %%

BOARD_INIT_GRID_STR = [
    ['黑车', '黑马', '黑相', '黑士', '黑帅', '黑士', '黑相', '黑马', '黑车'],
    [],
    ['', '黑炮', '', '', '', '', '', '黑炮', ''],
    ['黑兵', '', '黑兵', '', '黑兵', '', '黑兵', '', '黑兵'],
    [],
    [],
    ['红兵', '', '红兵', '', '红兵', '', '红兵', '', '红兵'],
    ['', '红炮', '', '', '', '', '', '红炮', ''],
    [],
    ['红车', '红马', '红相', '红士', '红帅', '红士', '红相', '红马', '红车'],
]

BOARD_INIT_GRID = [[None for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
for row, col in product(range(BOARD_HEIGHT), range(BOARD_WIDTH)):
  if BOARD_INIT_GRID_STR[row] and BOARD_INIT_GRID_STR[row][col]:
    BOARD_INIT_GRID[row][col] = Chess.from_str(BOARD_INIT_GRID_STR[row][col])


# %%


class Board:
  grid: list[list[Chess | None]]
  current_turn: ChessColor

  def __init__(self):
    self.grid = deepcopy(BOARD_INIT_GRID)
    self.current_turn = ChessColor.Red

  def __getitem__(self, pos: Position) -> Chess | None:
    """获取指定位置的棋子"""
    return self.grid[pos.row][pos.col]

  def __setitem__(self, pos: Position, chess: Chess | None):
    """设置指定位置的棋子"""
    self.grid[pos.row][pos.col] = chess

  def game_end(self) -> tuple[bool, ChessColor | None]:
    """
    检查游戏是否结束
    返回一个元组 (游戏结束, 胜利者)
    """
    red_king = any(chess for row in self.grid for chess in row if chess and chess.color ==
                   ChessColor.Red and chess.type == ChessType.King)
    black_king = any(chess for row in self.grid for chess in row if chess and chess.color ==
                     ChessColor.Black and chess.type == ChessType.King)

    if not red_king and not black_king:
      return True, None  # 平局
    elif not red_king:
      return True, ChessColor.Black  # 黑方胜利
    elif not black_king:
      return True, ChessColor.Red  # 红方胜利
    return False, None  # 游戏未结束

  # 当前回合所有可能的行动
  def available_actions(self) -> list[Action]:
    return []

  # 实际执行该动作 改变棋盘自身状态
  def do_action(self, action: Action):
    assert self[action.from_pos] == action.chess, f"{action.from_pos}位置预期是{action.chess} 实际为{self[action.from_pos]}"
    self[action.from_pos] = None
    self[action.to_pos] = action.chess
    self.current_turn = self.current_turn.next()  # 切换回合

  def do_move(self, move: Move):
    """执行一个移动"""
    chess = self[move.from_pos]
    assert chess is not None, f"从{move.from_pos}移动时没有棋子"
    action = Action(chess, move.from_pos, move.to_pos, self[move.to_pos])
    self.do_action(action)

  # 将棋盘状态转化为深度学习网络的输入
  def to_network_input(self) -> Tensor:
    return None

  def __str__(self):
    board_str = ""
    for row in range(BOARD_HEIGHT):
      for col in range(BOARD_WIDTH):
        chess = self.grid[row][col]
        if chess:
          board_str += str(chess)
        else:
          board_str += "  "
      board_str += "\n"
    return board_str


# %%
b = Board()
a = Action(Chess(ChessColor.Red, ChessType.Rook), Position(9, 0), Position(8, 0))

print(b)
b.do_action(a)
print(b)

# %%
