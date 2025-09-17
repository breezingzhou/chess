# %%
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning as L

from chess.utils import get_policy_train_data
from net.policy_net import PolicyNet

# %%
# 训练数据
states, move_probs = get_policy_train_data()
states_tensor = torch.stack(states)
move_probs_tensor = torch.stack(move_probs)
train_dataset = TensorDataset(states_tensor, move_probs_tensor)

# %%
# 训练
model = PolicyNet()
trainer = L.Trainer(max_epochs=1)
trainer.fit(model, train_dataloaders=DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=0))

# %%
# 加载已有模型
model = PolicyNet.load_from_checkpoint(
    "lightning_logs/version_12/checkpoints/epoch=9-step=2520.ckpt")
model.to('cpu')
# %%
from chess.board import Board
from chess.define import LEGAL_MOVES
b = Board()


# %%
model.eval()

input_tensor = b.to_network_input()
input_tensor = input_tensor.unsqueeze(0)
policy_logits = model(input_tensor)
index = policy_logits.argmax(dim=-1)
move = LEGAL_MOVES[index]
b.do_move(move)
b.to_image()

# %%
