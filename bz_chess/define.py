# %%
# import enum
# import aenum
# from aenum import MultiValueEnum
from enum import Enum, StrEnum, EnumMeta
from termcolor import colored
from torch import Tensor
import torch

BOARD_HEIGHT = 10
BOARD_WIDTH = 9

N_FEATURES = 7  # 7种棋子
# %%


# 执红先行
class ChessColor(Enum):
  Red = ('红', 1)
  Black = ('黑', -1)

  def next(self):
    return ChessColor.Red if self == ChessColor.Black else ChessColor.Black

  @property
  def number(self):
    return self.value[1]

  @classmethod
  def _missing_(cls, value: str | int):
    # ChessColor('红')/ ChessColor(1)的方式访问
    for member in cls:
      if member.value[0] == value or member.value[1] == value:
        return member
    return super()._missing_(value)


class ChessType(Enum):
  Rook = ('车', 0, 'r')
  Knight = ('马', 1, 'n')
  Bishop = ('相', 2, 'b')
  Advisor = ('士', 3, 'a')
  King = ('帅', 4, 'k')
  Cannon = ('炮', 5, 'c')
  Pawn = ('兵', 6, 'p')

  @property
  def display_name(self):
    return self.value[0]

  @property
  def tensor_index(self):
    return self.value[1]

  @property
  def short_name(self):
    return self.value[2]

  def __str__(self):
    return self.display_name

  def to_tensor(self) -> Tensor:
    tensor = torch.zeros(N_FEATURES, dtype=torch.float32)
    tensor[self.tensor_index] = 1
    return tensor

  @classmethod
  def _missing_(cls, value: str):
    # 支持通过ChessType('车')/ ChessType('r')的方式访问
    for member in cls:
      if member.value[0] == value or member.value[2] == value:
        return member
    return super()._missing_(value)


class Chess:
  color: ChessColor
  type: ChessType

  def __init__(self, color: ChessColor, type: ChessType):
    self.color = color
    self.type = type

  def __str__(self):
    color = "red" if self.color == ChessColor.Red else "blue"
    return colored(self.type.display_name, color)

  def __repr__(self):
    return f"{self.color.value}{self.type.display_name}"

  def __eq__(self, other):
    if not isinstance(other, Chess):
      return False
    return self.color == other.color and self.type == other.type

  def __ne__(self, other):
    return not self.__eq__(other)

  def to_tensor(self) -> Tensor:
    """将棋子转换为张量表示"""
    type_tensor = self.type.to_tensor()
    tensor = type_tensor * (1 if self.color == ChessColor.Red else -1)
    return tensor

  def to_fen(self) -> str:
    """将棋子转换为FEN字符串"""
    fen = self.type.short_name
    if self.color == ChessColor.Red:
      fen = fen.upper()
    return fen

  @classmethod
  def from_str(cls, s: str):
    if len(s) != 2:
      raise ValueError(f"Invalid chess string: {s}")
    color = ChessColor(s[0])
    type = ChessType(s[1])
    return cls(color, type)


# %%
chess = Chess.from_str("黑车")
chess.to_tensor()
# %%


class Position:
  row: int
  col: int

  def __init__(self, row: int, col: int):
    self.row = row
    self.col = col

  def flip(self):
    """旋转位置"""
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

  def to_move(self):
    """将动作转换为移动"""
    return Move(self.from_pos, self.to_pos)

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

  # "7747" 表示第一步红方将棋子从77移动到74（炮二平五）
  # 注意横纵的变化！！
  @classmethod
  def from_move_str(cls, move_str: str):
    if len(move_str) != 4:
      raise ValueError(f"Invalid move string: {move_str}")
    from_pos = Position(int(move_str[1]), int(move_str[0]))
    to_pos = Position(int(move_str[3]), int(move_str[2]))
    return cls(from_pos, to_pos)

  # "c3c4" 表示第一步红方将棋子从62移动到52（兵七进一）
  # 注意横纵的变化！！
  @classmethod
  def from_uci_str(cls, uci_str: str):
    if len(uci_str) != 4:
      raise ValueError(f"Invalid move string: {uci_str}")
    from_x = BOARD_HEIGHT - 1 - int(uci_str[1])
    to_x = BOARD_HEIGHT - 1 - int(uci_str[3])
    from_y = ord(uci_str[0]) - ord('a')
    to_y = ord(uci_str[2]) - ord('a')
    from_pos = Position(from_x, from_y)
    to_pos = Position(to_x, to_y)
    return cls(from_pos, to_pos)

  def to_move_str(self) -> str:
    """将移动转换为字符串表示"""
    return f"{self.from_pos.col}{self.from_pos.row}{self.to_pos.col}{self.to_pos.row}"

  def reverse(self):
    # 交换起始和目标位置
    return Move(self.to_pos, self.from_pos)

  def is_valid(self):
    return self.from_pos.is_valid() and self.to_pos.is_valid() and self.from_pos != self.to_pos

  def __eq__(self, other):
    if not isinstance(other, Move):
      return False
    return self.from_pos == other.from_pos and self.to_pos == other.to_pos

  def __ne__(self, other):
    return not self.__eq__(other)

  # Move作为dict的key需要实现__hash__和__eq__方法
  def __hash__(self):
    return hash(self.__str__())

  def __str__(self):
    return f"{self.from_pos}{self.to_pos}"

  def __repr__(self):
    return self.__str__()


# %%

def get_legal_moves() -> list[Move]:
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
  bishop_moves = [x for move in bishop_moves for x in (
      move, Move(move.from_pos.flip(), move.to_pos.flip()))]
  bishop_moves = [x for move in bishop_moves for x in (move, move.reverse())]

  # 马的移动方向
  knight_directions = [
      (-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1),
  ]

  legal_moves: list[Move] = []
  for row in range(BOARD_HEIGHT):
    for col in range(BOARD_WIDTH):
      src = Position(row, col)
      destinations = [Position(row, t) for t in range(BOARD_WIDTH)] + \
          [Position(t, col) for t in range(BOARD_HEIGHT)] + \
          [Position(row + a, col + b) for (a, b) in knight_directions]
      for dest in destinations:
        if dest.is_valid() and src != dest:
          legal_moves.append(Move(src, dest))

  legal_moves.extend(advisor_moves)
  legal_moves.extend(bishop_moves)
  legal_moves = sorted(legal_moves, key=lambda x: (
      x.from_pos.row, x.from_pos.col, x.to_pos.row, x.to_pos.col))
  return legal_moves


# %%
LEGAL_MOVES = get_legal_moves()

MOVE_TO_INDEX: dict[Move, int] = {move: i for i, move in enumerate(LEGAL_MOVES)}

MOVE_SIZE: int = len(LEGAL_MOVES)

# %%
from typing import TypeAlias

MoveTensor: TypeAlias = Tensor  # shape (MOVE_SIZE,), one-hot
StateTensor: TypeAlias = Tensor  # shape (N_FEATURES+1, BOARD_HEIGHT, BOARD_WIDTH)


def move_to_index_tensor(move: Move, mock_opponent: bool = False) -> MoveTensor:
  # 将移动转换为索引
  # mock_opponent表示是否为模拟对手的移动（即翻转棋盘）
  if mock_opponent:
    move = Move(move.from_pos.flip(), move.to_pos.flip())
  index = MOVE_TO_INDEX[move]
  tensor = torch.zeros(MOVE_SIZE, dtype=torch.float32)
  tensor[index] = 1.0
  return tensor
# %%


from dataclasses import dataclass


class ChessWinner(Enum):
  Red = ('红胜', 1)
  Black = ('黑胜', -1)
  Draw = ('和棋', 0)

  @classmethod
  def _missing_(cls, value: str | int):
    # ChessWinner('红胜')/ ChessWinner(1)的方式访问
    for member in cls:
      if member.value[0] == value or member.value[1] == value:
        return member
    return super()._missing_(value)

  @property
  def number(self):
    return self.value[1]


@dataclass
class ChessRecordData:
  red_player: str
  black_player: str
  winner: ChessWinner
  movelist: str


# 统一自我对局数据与大师对局数据的结构
@dataclass
class ChessRecord(ChessRecordData):
  id: int


# %%
