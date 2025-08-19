# %%
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from net import Net
from replay_chess_record import get_chess_train_data


class Training(L.LightningModule):
  def __init__(self, model, lr=0.001):
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
# %%

# 训练数据
states, move_probs = get_chess_train_data()
states_tensor = torch.stack(states)
move_probs_tensor = torch.stack(move_probs)
train_dataset = TensorDataset(states_tensor, move_probs_tensor)

# %%
# 训练
model = Training(Net())
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, train_dataloaders=DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=0))

# %%
# 加载已有模型
model = Training.load_from_checkpoint("lightning_logs/version_12/checkpoints/epoch=9-step=2520.ckpt", model=Net())
model.to('cpu')
# %%
from board import Board
from define import LEGAL_MOVES
b = Board()


# %%
model.eval()

input_tensor = b.to_network_input()
input_tensor = input_tensor.unsqueeze(0)
policy_logits, value = model(input_tensor)
index = policy_logits.argmax(dim=-1)
move = LEGAL_MOVES[index]
b.do_move(move)
b.to_image()
# %%
