# %%
# 两个agent对弈 获取棋谱 胜负关系等
from chess.board import Board
from chess.define import ChessColor, ChessWinner, Position, Action, Move
from players.base_player import BasePlayer
# %%


# %%


class Game:
  def __init__(self, red_player: BasePlayer, black_player: BasePlayer, debug: bool = False, evaluate: bool = False) -> None:
    self.red_player = red_player
    self.black_player = black_player
    self.debug = debug
    self.evaluate = evaluate
    if evaluate:
      self.red_player.evaluate = True
      self.black_player.evaluate = True
    self.board = Board()

    self.movelist: str = ""
    self.turns: int = 1

  def start_play_loop(self, draw_turns: int = 200) -> ChessWinner:
    # draw_turns 和棋的最大回合数
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
      elif self.turns >= draw_turns:
        return ChessWinner.Draw  # 和棋

  def log_evaluation(self) -> None:
    self.red_player.log_evaluation()
    self.black_player.log_evaluation()
