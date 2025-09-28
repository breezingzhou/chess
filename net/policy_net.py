# %%
import torch
import torch.nn as nn

from bz_chess.define import BOARD_HEIGHT, BOARD_WIDTH, MOVE_SIZE
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
    self.val_metrics = {}

  # 输出策略分布
  # 策略分布未经过softmax
  def forward(self, x):
    return self.model(x)

  # 策略损失

  def policy_loss(self, policy_logits, policy_true):
    policy_loss = nn.CrossEntropyLoss()(policy_logits, policy_true)
    return policy_loss

  # 返回loss 或者 dict中返回loss
  def training_step(self, batch, *, stage="train"):
    # TODO add value loss
    states, move_probs = batch
    policy_logits = self.forward(states)
    loss = self.policy_loss(policy_logits, move_probs)
    self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False)
    return loss

  def validation_step(self, batch):
    return self.training_step(batch, stage="val")

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    return optimizer

  def on_validation_epoch_end(self):
    if self.trainer.state.fn == "fit":
      self.val_metrics = self.trainer._logger_connector.metrics["log"]
      self.trainer._active_loop._logged_outputs.clear()  # type: ignore[attr-defined]
      self.trainer._active_loop._results.clear()  # type: ignore[attr-defined]

  def on_train_epoch_end(self):
    self.log_dict(self.val_metrics, on_epoch=True, on_step=False)
    self.val_metrics.clear()
