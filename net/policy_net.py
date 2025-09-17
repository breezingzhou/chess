# %%
import torch
import torch.nn as nn

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
    # 策略头
    self.policy_conv = nn.Sequential(
        nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=1, stride=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * BOARD_HEIGHT * BOARD_WIDTH, MOVE_SIZE),
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
    return policy_logits


# %%
import lightning as L


class PolicyNet(L.LightningModule):
  def __init__(self, model=Net(), lr=0.001):
    super().__init__()
    self.model = model
    self.learning_rate = lr
    # self.val_metrics = {}

  # 输出策略分布
  # 策略分布未经过softmax
  def forward(self, x):
    return self.model(x)

  # 策略损失

  def policy_loss(self, policy_logits, policy_true):
    policy_loss = nn.CrossEntropyLoss()(policy_logits, policy_true)
    return policy_loss

  # 返回loss 或者 dict中返回loss
  def training_step(self, batch):
    # TODO add value loss
    states, move_probs = batch
    policy_logits = self.forward(states)
    loss = self.policy_loss(policy_logits, move_probs)
    self.log("train_loss", loss)
    return loss

  # def train_dataloader(self):
  #   states, move_probs = get_chess_train_data()
  #   states_tensor = torch.stack(states)
  #   move_probs_tensor = torch.stack(move_probs)
  #   train_dataset = TensorDataset(states_tensor, move_probs_tensor)
  #   return DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    return optimizer
