# %%
from copy import deepcopy
from itertools import product
from pathlib import Path
from torch import Tensor
import torch
from PIL import Image, ImageDraw, ImageFont

# 如果 rs_chinese_chess 与 board.py 在同一包（同一目录下的模块），使用相对导入：
from .rs_chinese_chess import Board as RsBoard

from chess.define import N_FEATURES, Position, Chess, ChessType, ChessColor, Action, Move, BOARD_WIDTH, BOARD_HEIGHT, StateTensor
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

  # 只转化棋子
  def _to_fen(self) -> str:
    res = ""
    for row in self.grid:
      blank = 0
      row_str = ""
      for chess in row:
        if chess:
          if blank > 0:
            row_str += str(blank)
            blank = 0
          row_str += chess.to_fen()
        else:
          blank += 1
      if blank > 0:
        row_str += str(blank)
      res += row_str + "/"
    return res[:-1]  # 去掉最后的斜杠

  # rs engine中特殊的FEN格式
  def to_fen_rs(self) -> str:
    res = self._to_fen()
    res += " "
    res += "w" if self.current_turn == ChessColor.Red else "b"
    return res

  # 将棋盘状态转化为FEN字符串
  def to_fen(self) -> str:
    res = self._to_fen()
    res += " "
    res += "w" if self.current_turn == ChessColor.Red else "b"
    res += " - - 0 1"  # 这里简化了FEN字符串，实际应用中可能需要更详细的信息
    # 0 最近一次吃子或者进兵后棋局进行的步数(半回合数)，用来判断“50回合自然限着”；
    # 1 棋局的回合数。
    # todo Board保存回合数等信息
    return res

  def to_image(self) -> Image.Image:
    bd = BoardImageDisplay(self)
    return bd.display()

  # 当前回合所有可能的行动
  def available_actions(self) -> list[Action]:
    return []

  def available_moves(self) -> list[Move]:
    board = RsBoard.from_fen(self.to_fen_rs())
    moves = board.generate_move()
    return [Move(Position(m.pos_from.row, m.pos_from.col), Position(m.pos_to.row, m.pos_to.col)) for m in moves]

  # 实际执行该动作 改变棋盘自身状态
  def do_action(self, action: Action):
    assert self[action.from_pos] == action.chess, f"{action.from_pos}位置预期是{action.chess} 实际为{self[action.from_pos]}"
    self[action.from_pos] = None
    self[action.to_pos] = action.chess
    self.current_turn = self.current_turn.next()  # 切换回合

  def do_move(self, move: Move, check: bool = False):
    """执行一个移动"""
    # chess = self[move.from_pos]
    # assert chess is not None, f"从{move.from_pos}移动时没有棋子"
    # assert chess.color == self.current_turn, f"当前是{self.current_turn}回合 不能移动{chess.color}棋子"
    if check:
      available_moves = self.available_moves()
      assert move in available_moves, f"移动{move}不合法"
    chess = self[move.from_pos]
    action = Action(chess, move.from_pos, move.to_pos, self[move.to_pos])
    self.do_action(action)

  # 将棋盘状态转化为深度学习网络的输入
  def to_network_input(self) -> StateTensor:
    # TODO 增加上一个回合对手棋子的移动信息
    tensor = torch.zeros((N_FEATURES, BOARD_HEIGHT, BOARD_WIDTH), dtype=torch.float32)
    for row in range(BOARD_HEIGHT):
      for col in range(BOARD_WIDTH):
        chess = self.grid[row][col]
        if chess:
          tensor[:, row, col] = chess.to_tensor()
    # 添加当前回合信息
    layer = torch.ones((1, BOARD_HEIGHT, BOARD_WIDTH), dtype=torch.float32)
    layer *= 1 if self.current_turn == ChessColor.Red else -1
    return torch.cat((tensor, layer), dim=0)

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
move = Move(Position(9, 0), Position(8, 0))
move in b.available_moves()
#%%
print(b)
print(b.to_fen())
b.do_action(a)
print(b)
print(b.to_fen())
t = b.to_network_input()
moves = b.available_moves()

# %%


class BoardImageDisplay:
  def __init__(self, board: Board, margin: int = 20, cell_size: int = 40):
    self.board = board
    self.margin = margin
    self.cell_size = cell_size

    # 十字线 间隔 和 长度
    self.crosshairs_interval = 3
    self.crosshairs_length = 7

    self.total_width = (BOARD_WIDTH - 1) * cell_size + 2 * margin
    self.total_height = (BOARD_HEIGHT - 1) * cell_size + 2 * margin
    # 计算边界
    self.left = margin
    self.right = self.total_width - margin
    self.top = margin
    self.bottom = self.total_height - margin
    # 中间线
    self.top_middle = margin + ((BOARD_HEIGHT - 1) // 2) * cell_size
    self.bottom_middle = self.top_middle + cell_size
    # 字体初始化
    font_path = Path(__file__).parent / "res/simhei.ttf"
    self.font = ImageFont.truetype(font_path, 24)
    # 创建图像和绘图对象
    self.img = Image.new("RGB", (self.total_width, self.total_height), (185, 102, 47))
    self.draw = ImageDraw.Draw(self.img)

  def position_to_pixel(self, pos: Position) -> tuple[int, int]:
    """将棋盘位置转换为像素坐标"""
    x = self.left + pos.col * self.cell_size
    y = self.top + pos.row * self.cell_size
    return x, y

  def display(self) -> Image.Image:
    self.draw_board()
    self.draw_chess()
    return self.img

  def draw_chess(self):
    for row in range(BOARD_HEIGHT):
      for col in range(BOARD_WIDTH):
        chess = self.board.grid[row][col]
        if chess:
          # 画棋子
          self._draw_chess(chess, Position(row, col))

  def _draw_chess(self, chess: Chess, pos: Position):
    x, y = self.position_to_pixel(pos)
    # 绘制棋子图标
    self.draw.circle((x, y), radius=self.cell_size // 2 - 2, fill=(243, 172, 87), outline="black")
    # 绘制棋子文字
    text = chess.type.display_name
    _, _, text_width, text_height = self.draw.textbbox((0, 0), text, font=self.font)

    fill_color = "red" if chess.color == ChessColor.Red else "black"
    self.draw.text((x - text_width // 2, y - text_height // 2),
                   text, fill=fill_color, font=self.font)

  # 画基础棋盘
  def draw_board(self):
    # 横线
    for i in range(BOARD_HEIGHT):
      y = self.margin + i * self.cell_size
      self.draw.line([(self.left, y), (self.right, y)], fill="black", width=2)
    # 竖线
    self.draw.line([(self.left, self.top), (self.left, self.bottom)], fill="black", width=2)
    self.draw.line([(self.right, self.top), (self.right, self.bottom)], fill="black", width=2)
    for i in range(1, BOARD_WIDTH):
      x = self.margin + i * self.cell_size
      self.draw.line([(x, self.top), (x, self.top_middle)], fill="black", width=2)
      self.draw.line([(x, self.bottom_middle), (x, self.bottom)], fill="black", width=2)
    # 士的斜线
    advisor_lines = [
        (Position(0, 3), Position(2, 5)),
        (Position(0, 5), Position(2, 3)),
        (Position(9, 3), Position(7, 5)),
        (Position(9, 5), Position(7, 3)),
    ]
    for pos1, pos2 in advisor_lines:
      x1, y1 = self.position_to_pixel(pos1)
      x2, y2 = self.position_to_pixel(pos2)
      self.draw.line([(x1, y1), (x2, y2)], fill="black", width=2)
    # 炮、兵初始位置的十字线
    positions = [
        # 炮
        Position(2, 1), Position(2, 7), Position(7, 1), Position(7, 7),
        # 兵
        Position(3, 0), Position(3, 2), Position(3, 4), Position(3, 6), Position(3, 8),
        Position(6, 0), Position(6, 2), Position(6, 4), Position(6, 6), Position(6, 8),
    ]
    for pos in positions:
      x, y = self.position_to_pixel(pos)
      for direction in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        dx, dy = direction
        start_x = x + dx * self.crosshairs_interval
        start_y = y + dy * self.crosshairs_interval
        if start_x < self.left or start_x > self.right:
          continue
        self.draw.line([(start_x, start_y), (start_x, start_y + dy *
                       self.crosshairs_length)], fill="black", width=1)
        self.draw.line([(start_x, start_y), (start_x + dx * self.crosshairs_length, start_y)],
                       fill="black", width=1)


# %%
board = Board()
bd = BoardImageDisplay(board)
bd.display()
a = Action(Chess(ChessColor.Red, ChessType.Cannon), Position(7, 1), Position(7, 2))
board.do_action(a)
bd = BoardImageDisplay(board)
bd.display()

# %%
