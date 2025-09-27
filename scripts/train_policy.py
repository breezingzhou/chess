# %%
from _common import *
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
import lightning as L
import logging
from typing import Optional, cast, Tuple


from net.policy_net import PolicyNet
from utils.common import split_dataset, PolicyCheckPointDir
from utils.db.loader import get_policy_train_data
from utils.utils import setup_logging
# %%


class PolicyTrain:
  chess_record_num: int
  max_epochs: int
  ###

  def __init__(self, policy_model: Optional[PolicyNet] = None, checkpoint_dir=PolicyCheckPointDir, chess_record_num: int = 0, max_epochs=100):
    self.chess_record_num = chess_record_num
    self.max_epochs = max_epochs
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_path = self.checkpoint_dir / f"base.ckpt"
    ###
    self.policy_model = policy_model or PolicyNet()

  def _load_train_data(self):
    cache_path = self.checkpoint_dir / f"train_data_cache_{self.chess_record_num}.pt"

    if cache_path.exists():
      logging.info(f"加载已有的训练数据缓存 {cache_path}")
      states_tensor, move_probs_tensor = cast(
          Tuple[Tensor, Tensor], torch.load(cache_path))
      return states_tensor, move_probs_tensor

    logging.info(f"生成训练数据")
    states_tensor, move_probs_tensor = get_policy_train_data(
        chess_record_num=self.chess_record_num, version=0)
    logging.info(f"训练数据生成完毕，缓存到 {cache_path}")
    torch.save((states_tensor, move_probs_tensor), cache_path)
    return states_tensor, move_probs_tensor

  def train_model(self):
    logging.info(f"开始训练策略网络")
    states_tensor, move_probs_tensor = self._load_train_data()

    full_dataset = TensorDataset(states_tensor, move_probs_tensor)
    train_dataset, val_dataset = split_dataset(full_dataset)

    trainer = L.Trainer(max_epochs=self.max_epochs, default_root_dir=WORKSPACE)
    trainer.fit(
        self.policy_model,
        train_dataloaders=DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0),
        val_dataloaders=DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    )

    logging.info(f"训练完成 模型保存路径:{self.checkpoint_path}")
    trainer.save_checkpoint(self.checkpoint_path)
    logging.info("保存模型完成")


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


if __name__ == "__main__":
  setup_logging()
  checkpoint_path = WORKSPACE / "lightning_logs/version_42/checkpoints/epoch=20-step=243012.ckpt"
  policy_model = PolicyNet.load_from_checkpoint(checkpoint_path)
  trainer = PolicyTrain(policy_model=policy_model, chess_record_num=0, max_epochs=100)
  trainer.train_model()
# %%
