# %%
# 只通过策略网络来进行落子
import torch
from torch import Tensor
import torch.nn.functional as F

from .base_player import BasePlayer
from chess.board import Board
from chess.define import LEGAL_MOVES, MOVE_TO_INDEX, Move, StateTensor
from net.policy_net import PolicyNet
# %%


class PolicyPlayer(BasePlayer):
  def __init__(self, name: str, model: PolicyNet, is_selfplay: bool = True):
    super().__init__(name)
    self.is_selfplay = is_selfplay  # 自我对弈时候按照概率采样 否则选取概率最大值
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

    if self.is_selfplay:
      probs = F.softmax(policy_logits, dim=-1)
      index = torch.multinomial(probs, num_samples=1, replacement=True)
    else:
      index = policy_logits.argmax(dim=-1)
    return legal_moves[index]
