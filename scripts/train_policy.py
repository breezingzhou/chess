# %%
from _common import *
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning as L

from net.policy_net import PolicyNet
from utils.db.loader import get_policy_train_data

# %%
# 训练数据
states, move_probs = get_policy_train_data(chess_record_num=5000, version=0)
states_tensor = torch.stack(states)
move_probs_tensor = torch.stack(move_probs)
train_dataset = TensorDataset(states_tensor, move_probs_tensor)

# %%
# 训练
model = PolicyNet()
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, train_dataloaders=DataLoader(
    train_dataset, batch_size=256, shuffle=True, num_workers=0))

# %%
# 加载已有模型
model = PolicyNet.load_from_checkpoint(
    "lightning_logs/version_15/checkpoints/epoch=13-step=3584.ckpt")

# %%
from chess.board import Board
from chess.define import LEGAL_MOVES


def test_model(model: PolicyNet):
  model.to('cpu')
  model.eval()

  b = Board()
  images = [b.to_image()]
  for _ in range(10):
    input_tensor = b.to_network_input()
    input_tensor = input_tensor.unsqueeze(0)
    policy_logits = model(input_tensor)
    index = policy_logits.argmax(dim=-1)
    move = LEGAL_MOVES[index]
    b.do_move(move)
    images.append(b.to_image())

# %%
