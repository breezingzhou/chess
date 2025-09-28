# %%
# 只通过策略网络来进行落子
import torch
from torch import Tensor
import torch.nn.functional as F

from .base_player import BasePlayer
from bz_chess.board import Board
from bz_chess.define import LEGAL_MOVES, MOVE_TO_INDEX, Move, StateTensor
from net.policy_net import PolicyNet
# %%
from dataclasses import dataclass
import logging


@dataclass
class TopKEvaluate:
  top1: int = 0
  top3: int = 0
  top10: int = 0
  total: int = 0

# %%


class PolicyPlayer(BasePlayer):
  def __init__(self, name: str, model: PolicyNet, temperature: float = 1.0, evaluate: bool = False, device: str = "cuda") -> None:
    super().__init__(name, evaluate=evaluate)
    # 温度系数 探索 vs 利用
    # temperature = 1 正常的softmax
    # temperature 越高 越随机
    # 额外逻辑 temperature = 0 直接贪心
    self.temperature = temperature
    self.model = model
    self.device = device
    self.model.to(device)
    self.model.eval()
    # TODO 是不是在gpu上推断更快
    self.topk = TopKEvaluate()

  @property
  def display_name(self) -> str:
    return f"{self.name}_t={self.temperature:.2f}"

  def infer(self, state: StateTensor) -> Tensor:
    batch = state.unsqueeze(0)
    batch_device = batch.to(self.device)
    policy_logits: Tensor = self.model(batch_device)  # 获取每个走法的概率
    return policy_logits[0].to("cpu")

  def get_move(self, board: Board) -> Move:
    state = board.to_network_input()
    policy_logits: Tensor = self.infer(state)  # 获取每个走法的概率

    legal_moves = board.available_moves()
    legal_moves_ids = [MOVE_TO_INDEX[m] for m in legal_moves]
    legal_policy_logits = policy_logits[legal_moves_ids]

    if self.evaluate:
      self.top_k_evaluate(legal_policy_logits, policy_logits)

    # 直接返回最大概率的走法
    if self.temperature == 0:
      index = legal_policy_logits.argmax(dim=-1)
    else:
      logits = legal_policy_logits / self.temperature
      probs = F.softmax(logits, dim=-1)
      index = torch.multinomial(probs, num_samples=1, replacement=True)
    return legal_moves[index]

  # 统计一下legal_policy_logits中最大值 在policy_logits中前1 前3 前10的比例
  def top_k_evaluate(self, legal_policy_logits: Tensor, policy_logits: Tensor) -> None:
    legal_max = legal_policy_logits.max().item()
    sorted, indices = torch.sort(policy_logits, descending=True)
    top1 = sorted[0].item()
    top3 = sorted[2].item()
    top10 = sorted[9].item()
    self.topk.total += 1
    if legal_max == top1:
      self.topk.top1 += 1
    if legal_max >= top3:
      self.topk.top3 += 1
    if legal_max >= top10:
      self.topk.top10 += 1

  def log_evaluation(self) -> None:
    if not (self.evaluate and self.topk.total > 0):
      return
    top1_prob = self.topk.top1 / self.topk.total
    top3_prob = self.topk.top3 / self.topk.total
    top10_prob = self.topk.top10 / self.topk.total
    logging.info(
        f"[最优合法走法 排名占比] [当前温度: {self.temperature}]  top1: {top1_prob:.4f}, top3: {top3_prob:.4f}, top10: {top10_prob:.4f}")
