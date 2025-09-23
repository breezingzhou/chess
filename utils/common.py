# %%
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent

CheckPointDir = WORKSPACE / "res/checkpoints"
PolicyCheckPointDir = CheckPointDir / "policy"
ValueCheckPointDir = CheckPointDir / "value"
# base.ckpt 是使用大师对局数据训练的模型 epoch=46-step=49585
# 在train_policy.py中生成
# 手动复制到 PolicyCheckPointDir 下


# %%
def cal_log_epoch(total: int) -> int:
  return max(total // 10, 1)


# %%
from torch.utils.data import TensorDataset, Subset, random_split
import torch


def split_dataset(dataset: TensorDataset, train_rate: float = 0.8) -> tuple[Subset, Subset]:
  total_size = len(dataset)
  train_size = int(train_rate * total_size)
  val_size = total_size - train_size

  train_dataset, val_dataset = random_split(
      dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
  return train_dataset, val_dataset
