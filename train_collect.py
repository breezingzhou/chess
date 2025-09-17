# %%
# 加载模型自我对弈 生成对弈数据 供下一次迭代
# 输入是当前轮次的模型  输出是当前模型下对弈的数据
from board import Board


class CollectSelfPlayData:
  def __init__(self, model):
    self.model = model

  # 自我对弈
  def self_play(self):
    count = 0
    board = Board()
    while True:
      count += 1


# %%
from define import MOVE_TO_INDEX, Move
import torch.nn as nn
import torch


class BoardEvaluate:
  def __init__(self, model: nn.Module):
    self.model = model
    self.model.to('cpu')

  def policy_value_fn(self, board: Board) -> tuple[list[tuple[Move, float]], float]:
    self.model.eval()
    input_tensor = board.to_network_input()
    input_tensor = input_tensor.unsqueeze(0)
    policy_logits, value = self.model(input_tensor)
    # 过滤合法的操作
    legal_moves = board.available_moves()
    legal_moves_ids = [MOVE_TO_INDEX[m] for m in legal_moves]

    legal_policy_logits = policy_logits[:, legal_moves_ids]
    legal_policy_probs = torch.softmax(legal_policy_logits, dim=-1)
    legal_policy_probs = legal_policy_probs.tolist()[0]

    return list(zip(legal_moves, legal_policy_probs)), value.item()


# %%
# from mcts import MCTS, MCTSPlayer
# from net import PolicyValueNet
# import torch
# model = PolicyValueNet.load_from_checkpoint(
#     "lightning_logs/version_12/checkpoints/epoch=9-step=2520.ckpt")
# be = BoardEvaluate(model)


# # %%
# mcts = MCTS(be.policy_value_fn)
# mcts._playout(Board())
# %%
