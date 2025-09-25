# %%

from chess.board import Board
from chess.define import Move
from players.base_player import BasePlayer
from net.value_net import ValueNet
from net.policy_net import PolicyNet
from utils.mcts import MCTS


class MCTSPlayer(BasePlayer):
  def __init__(self, name: str, policy_net: PolicyNet, value_net: ValueNet, temperature: float = 1.0, c_puct: int = 5, n_simulations: int = 1000, evaluate: bool = False, device: str = "cuda"):
    super().__init__(name)
    self.policy_net = policy_net
    self.value_net = value_net
    self.c_puct = c_puct
    self.n_simulations = n_simulations
    self.temperature = temperature
    self.evaluate = evaluate
    self.device = device
    #
    self.policy_net.to(device)
    self.value_net.to(device)
    self.policy_net.eval()
    self.value_net.eval()

  @property
  def display_name(self) -> str:
    return f"{self.name}_c_puct={self.c_puct:.2f}_n_simulations={self.n_simulations}"

  def get_move(self, board: Board) -> Move:
    mcts = MCTS(self.policy_net, self.value_net, self.c_puct, device=self.device)
    root = mcts.search(board, n_simulations=self.n_simulations, add_dirichlet=False)
    # policy = mcts.get_policy(root, temperature=self.temperature)
    move = mcts.select_move(root, temperature=self.temperature)
    return move
