# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


from chess.define import BOARD_HEIGHT, BOARD_WIDTH, MOVE_SIZE
from net.res_block import ResBlock
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

    # 价值头
    value = self.value_conv(x)
    value = F.tanh(value)

    return value


# %%
import lightning as L


class ValueNet(L.LightningModule):
  def __init__(self, model=Net(), lr=0.001):
    super().__init__()
    self.model = model
    self.learning_rate = lr
    # self.val_metrics = {}

  # 输出价值 [-inf, inf]
  def forward(self, x):
    return self.model(x)

  # 价值损失
  def value_loss(self, value, value_true):
    value_loss = F.mse_loss(input=value, target=value_true)
    return value_loss

  # 返回loss 或者 dict中返回loss
  def training_step(self, batch):
    # TODO generate batch
    state, value_true = batch
    value = self.forward(state)
    loss = self.value_loss(value, value_true)
    self.log("train_loss", loss)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    return optimizer
