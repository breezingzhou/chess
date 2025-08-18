# %%
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from define import BOARD_HEIGHT, BOARD_WIDTH, MOVE_SIZE
# %%


class ResBlock(nn.Module):

  def __init__(self, num_filters=256):
    super().__init__()

    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=num_filters, out_channels=num_filters,
                  kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(num_filters),
        nn.ReLU()
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=num_filters, out_channels=num_filters,
                  kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(num_filters),
    )

  def forward(self, x):
    y = self.conv1(x)
    y = self.conv2(y)
    y += x
    return nn.ReLU()(y)

# %%


class Net(nn.Module):
  def __init__(self, num_channels=256, num_res_blocks=7):
    super().__init__()

    # 初始化特征
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels=8, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU()
    )
    # 残差块抽取特征
    self.res_blocks = nn.ModuleList(
        [ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)]
    )
    # 策略头
    self.policy_conv = nn.Sequential(
        nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=1, stride=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * BOARD_HEIGHT * BOARD_WIDTH, MOVE_SIZE),
    )
    # 价值头
    self.value_conv = nn.Sequential(
        nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=(1, 1), stride=(1, 1)),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * BOARD_HEIGHT * BOARD_WIDTH, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )

  def forward(self, x):
    """
    x: [batch_size, 8, BOARD_HEIGHT, BOARD_WIDTH]
    """
    # 公共头
    x = self.conv(x)
    for res_block in self.res_blocks:
      x = res_block(x)

    # 策略头
    policy_logits = self.policy_conv(x)
    # policy_probs = F.softmax(policy_logits, dim=1)

    # 价值头
    value = self.value_conv(x)
    # value = F.tanh(value)

    return policy_logits, value


# %%
import torch
from board import Board
from define import Move


class PolicyValueNet():
  def __init__(self):
    self.policy_value_net = Net()
    self.optimizer = torch.optim.Adam(self.policy_value_net.parameters(), lr=0.001)
    self.loss_fn = nn.MSELoss()

  def policy_value_fn(self, board: Board):
    """
    接收board的盘面状态，返回落子概率和盘面评估得分
    """
    self.policy_value_net.eval()  # 设置为评估模式
    input_tensor = board.to_network_input()
    policy_probs, value = self.policy_value_net(input_tensor)
    return None
