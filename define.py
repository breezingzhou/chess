# %%
from enum import Enum, IntEnum, StrEnum
from termcolor import colored

BOARD_WIDTH = 9
BOARD_HEIGHT = 10

# %%


# 执红先行
class ChessColor(StrEnum):
  Red = '红'
  Black = '黑'

  def next(self):
    return ChessColor.Red if self == ChessColor.Black else ChessColor.Black


class ChessType(StrEnum):
  King = '帅'     #
  Advisor = '士'  #
  Bishop = '相'   #
  Knight = '马'   #
  Rook = '车'     #
  Cannon = '炮'   #
  Pawn = '兵'     #


class Chess:
  color: ChessColor
  type: ChessType

  def __init__(self, color: ChessColor, type: ChessType):
    self.color = color
    self.type = type

  def __str__(self):
    color = "red" if self.color == ChessColor.Red else "blue"
    return colored(self.type.value, color)

  def __repr__(self):
    return f"{self.color.value}{self.type.value}"

  def __eq__(self, other):
    if not isinstance(other, Chess):
      return False
    return self.color == other.color and self.type == other.type

  def __ne__(self, other):
    return not self.__eq__(other)

  @classmethod
  def from_str(cls, s: str):
    if len(s) != 2:
      raise ValueError(f"Invalid chess string: {s}")
    color = ChessColor(s[0])
    type = ChessType(s[1])
    return cls(color, type)


# %%


class Position:
  row: int
  col: int

  def __init__(self, row: int, col: int):
    self.row = row
    self.col = col

  def flip(self):
    """翻转位置"""
    return Position(BOARD_HEIGHT - 1 - self.row, BOARD_WIDTH - 1 - self.col)

  def is_valid(self):
    """检查位置是否在棋盘内"""
    return 0 <= self.row < BOARD_HEIGHT and 0 <= self.col < BOARD_WIDTH

  def __eq__(self, other):
    if not isinstance(other, Position):
      return False
    return self.row == other.row and self.col == other.col

  def __ne__(self, other):
    return not self.__eq__(other)

  @property
  def pretty(self):
    return f"{chr(ord('a') + self.col)}{chr(ord('0') + BOARD_HEIGHT - 1 - self.row)}"

  def __str__(self):
    return f"[{self.row}, {self.col}]"

  def __repr__(self):
    return self.__str__()


# %%
p = Position(0, 0)
c = ChessColor.Red
c.next().next()
# %%


class Action:
  chess: Chess
  from_pos: Position
  to_pos: Position
  capture: Chess | None = None

  def __init__(self, chess: Chess, from_pos: Position, to_pos: Position, capture: Chess | None = None):
    self.chess = chess
    self.from_pos = from_pos
    self.to_pos = to_pos
    self.capture = capture

  def __str__(self):
    ext = f" (eat {self.capture})" if self.capture else ""
    return f"Action {self.chess} from {self.from_pos} to {self.to_pos}{ext}"

  def __repr__(self):
    return self.__str__()


# %%
class Move:
  from_pos: Position
  to_pos: Position

  def __init__(self, from_pos: Position, to_pos: Position):
    self.from_pos = from_pos
    self.to_pos = to_pos

  def reverse(self):
    return Move(self.to_pos, self.from_pos)

  def is_valid(self):
    return self.from_pos.is_valid() and self.to_pos.is_valid() and self.from_pos != self.to_pos

  def __str__(self):
    return f"{self.from_pos}{self.to_pos}"

  def __repr__(self):
    return self.__str__()


# %%
# 士的移动
advisor_pos = [Position(0, 3), Position(0, 5), Position(1, 4), Position(2, 3), Position(2, 5)]
advisor_moves = [
    Move(advisor_pos[0], advisor_pos[2]),
    Move(advisor_pos[1], advisor_pos[2]),
    Move(advisor_pos[3], advisor_pos[2]),
    Move(advisor_pos[4], advisor_pos[2]),
]
advisor_moves = [x for move in advisor_moves for x in (
    move, Move(move.from_pos.flip(), move.to_pos.flip()))]
advisor_moves = [x for move in advisor_moves for x in (move, move.reverse())]

# 相的移动
bishop_pos = [Position(0, 2), Position(0, 6), Position(2, 0), Position(2, 4),
              Position(2, 8), Position(4, 2), Position(4, 6)]
bishop_moves = [
    Move(bishop_pos[0], bishop_pos[2]),
    Move(bishop_pos[0], bishop_pos[3]),
    Move(bishop_pos[1], bishop_pos[3]),
    Move(bishop_pos[1], bishop_pos[4]),
    Move(bishop_pos[5], bishop_pos[2]),
    Move(bishop_pos[5], bishop_pos[3]),
    Move(bishop_pos[6], bishop_pos[3]),
    Move(bishop_pos[6], bishop_pos[4]),
]
bishop_moves = [x for move in advisor_moves for x in (
    move, Move(move.from_pos.flip(), move.to_pos.flip()))]
bishop_moves = [x for move in advisor_moves for x in (move, move.reverse())]

# 马的移动方向
knight_directions = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1),
]

moves = []
for row in range(BOARD_HEIGHT):
  for col in range(BOARD_WIDTH):
    src = Position(row, col)
    destinations = [Position(row, t) for t in range(BOARD_WIDTH)] + \
                   [Position(t, col) for t in range(BOARD_HEIGHT)] + \
                   [Position(row + a, col + b) for (a, b) in knight_directions]
    for dest in destinations:
      if dest.is_valid() and src != dest:
        moves.append(Move(src, dest))

# %%
