# %%
from _common import *
# 加载模型自我对弈 生成对弈数据 并保存到数据库
# 从数据库加载对弈数据 训练策略模型
import logging
import lightning as L
import torch
from torch.utils.data import TensorDataset, DataLoader

from utils import show_images_in_slider, setup_logging, collect_selfplay_data, WORKSPACE
from utils.db import SelfPlayChessRecordDAL, SelfPlayChessRecord
from utils.db.loader import get_policy_train_data, get_selfplay_chess_records
from chess import ChessRecordData, Game
from net.policy_net import PolicyNet
from players.policy_player import PolicyPlayer

# %%

# %%


# %%


# %%


class SelfPlayTrainLoop:
  checkpoint_dir: Path
  base_model: PolicyNet
  model_version: int

  def __init__(self, checkpoint_dir=PolicyCheckPointDir):
    self.checkpoint_dir = checkpoint_dir
    self.base_model = PolicyNet.load_from_checkpoint(checkpoint_dir / "base.ckpt")

    self.model_version = self._get_current_model_version()
    self.data_version = self.model_version + 1
    self.current_model = self._load_model(self.model_version)

  def _get_current_model_version(self) -> int:
    # 遍历 PolicyCheckPointDir 下的所有模型文件 获取最大的版本号
    files = self.checkpoint_dir.glob("version_*.ckpt")
    versions = [int(f.stem.split("_")[1]) for f in files]
    return max(versions) if versions else 0

  def _load_model(self, version: int) -> PolicyNet:
    if version == 0:
      return PolicyNet.load_from_checkpoint(self.checkpoint_dir / "base.ckpt")
    path = self.checkpoint_dir / f"version_{version}.ckpt"
    model = PolicyNet.load_from_checkpoint(path)
    return model

  def _gen_enough_selfplay_data(self, train_record_limit: int = 500, num_games: int = 100):
    # num_games 表示每次进行自我对弈的局数
    # train_record_limit 表示需要生成多少条有效的自我对弈数据后才进行训练
    # 使用模型生成足够的自我对弈数据
    while True:
      train_records = get_selfplay_chess_records(version=self.data_version)
      if len(train_records) >= train_record_limit:
        logging.info(
            f"自我对弈数据已足够 当前数据版本:{self.data_version} 共有 {len(train_records)} 条")
        return
      logging.info(
          f"自我对弈数据不足 当前数据版本:{self.data_version} 仅有 {len(train_records)} 条 需要{train_record_limit} 条 继续使用最新模型进行自我对弈")

      red_player = PolicyPlayer("红方", model=self.current_model, temperature=2.0)
      black_player = PolicyPlayer("黑方", model=self.base_model, temperature=2.0)
      collect_selfplay_data(red_player, black_player, num_games=num_games,
                            version=self.data_version, draw_turns=200)

  def _train_model(self, max_epochs=100):
      # 加载最新的对弈数据
    states_tensor, move_probs_tensor = get_policy_train_data(
        version=self.data_version, chess_record_num=None)

    train_dataset = TensorDataset(states_tensor, move_probs_tensor)
    # 训练模型
    logging.info(f"开始训练模型 当前数据版本:{self.data_version} 共有 {len(train_dataset)} 条训练数据")
    model = self.current_model
    trainer = L.Trainer(max_epochs=max_epochs, default_root_dir=WORKSPACE)
    trainer.fit(model, train_dataloaders=DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=0))
    # 保存最新的模型
    model_save_path = self.checkpoint_dir / f"version_{self.data_version}.ckpt"
    logging.info(f"训练完成 保存最新模型 保存路径:{model_save_path}")
    trainer.save_checkpoint(model_save_path)
    logging.info("保存模型完成")

  def selfplay_train_loop(self, turns: int, train_record_limit: int = 500, max_epochs: int = 100):
    # version_0_model => version_1_data => version_1_model
    # train_record_limit 表示需要生成多少条有效的自我对弈数据后才进行训练
    # turns 表示进行多少轮自我对弈+训练
    # max_epochs 表示每轮训练的最大epoch数

    logging.info(f"开始自我对弈+训练 循环 {turns} 轮 每轮生成至少 {train_record_limit} 条自我对弈数据")

    for turn in range(turns):
      logging.info(f"开始第 {turn + 1} / {turns} 轮自我对弈+训练")
      logging.info(f"当前模型版本:{self.model_version} 数据版本:{self.data_version}")
      # 自我对弈 生成数据
      self._gen_enough_selfplay_data(
          train_record_limit=train_record_limit)

      self._train_model(max_epochs=max_epochs)
      # 更新模型版本
      self.model_version += 1
      self.data_version = self.model_version + 1
      # self.current_model = self._load_model(self.model_version)


# %%
# %%
if __name__ == "__main__":
  setup_logging()
  train = SelfPlayTrainLoop(checkpoint_dir=PolicyCheckPointDir)
  train.selfplay_train_loop(turns=5, train_record_limit=500, max_epochs=100)

# %%
