# %%
# 两个agent对弈 获取棋谱 胜负关系等
from board import Board
from define import ChessColor, Position, Action, Move
# %%


class Player:
  def __init__(self, name: str):
    self.name = name

  def get_move(self, board: Board) -> Move:
    raise NotImplementedError
# %%


class Game:
  def __init__(self, red_player: Player, black_player: Player, tie_turns: int = 150, debug: bool = False):
    self.red_player = red_player
    self.black_player = black_player
    self.tie_turns = tie_turns  # 和棋的最大回合数
    self.debug = debug
    self.board = Board()

    self.movelist: str = ""
    self.turns: int = 1

  def start_play_loop(self) -> ChessColor | None:
    while True:
      current_player = self.red_player if self.board.current_turn == ChessColor.Red else self.black_player
      move = current_player.get_move(self.board)
      if self.debug:
        print(f"当前回合 {self.turns} 轮到 {self.board.current_turn}方 {current_player.name} 走")
        print(f"走法: {move} ")
      self.board.do_move(move)
      self.movelist += move.to_move_str()
      self.turns += 1

      game_end, winner = self.board.game_end()
      if game_end:
        return winner
      elif self.turns >= self.tie_turns:
        return None  # 和棋
