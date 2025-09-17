# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from chess.define import BOARD_HEIGHT, BOARD_WIDTH, MOVE_SIZE
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
    value = F.tanh(value)

    return policy_logits, value


# %%
import lightning as L


class PolicyValueNet(L.LightningModule):
  def __init__(self, model=Net(), lr=0.001):
    super().__init__()
    self.model = model
    self.learning_rate = lr
    # self.val_metrics = {}

  # 输出策略分布 和 价值
  # 策略分布未经过softmax 价值[-inf, inf]
  def forward(self, x):
    return self.model(x)

  # 价值损失
  def value_loss(self, value, value_true):
    value_loss = F.mse_loss(input=value, target=value_true)
    return value_loss

  # 策略损失
  def policy_loss(self, policy_logits, policy_true):
    policy_loss = nn.CrossEntropyLoss()(policy_logits, policy_true)
    return policy_loss

  # 返回loss 或者 dict中返回loss
  def training_step(self, batch):
    # TODO add value loss
    states, move_probs = batch
    policy_logits, value = self.forward(states)
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
