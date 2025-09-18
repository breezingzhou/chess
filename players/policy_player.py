# %%
# 只通过策略网络来进行落子
from torch import Tensor
from .base_player import BasePlayer
from chess.board import Board
from chess.define import LEGAL_MOVES, MOVE_TO_INDEX, Move, StateTensor
from net.policy_net import PolicyNet
# %%


class PolicyPlayer(BasePlayer):
  def __init__(self, name: str, model: PolicyNet):
    super().__init__(name)
    self.model = model
    self.model.to('cpu')
    # TODO 是不是在gpu上推断更快

  def infer(self, state: StateTensor) -> Tensor:
    self.model.eval()
    batch = state.unsqueeze(0)
    policy_logits: Tensor = self.model(batch)  # 获取每个走法的概率
    return policy_logits[0]

  def get_move(self, board: Board) -> Move:
    state = board.to_network_input()
    policy_logits: Tensor = self.infer(state)  # 获取每个走法的概率

    legal_moves = board.available_moves()
    legal_moves_ids = [MOVE_TO_INDEX[m] for m in legal_moves]

    policy_logits = policy_logits[legal_moves_ids]
    index = policy_logits.argmax(dim=-1)
    return legal_moves[index]
