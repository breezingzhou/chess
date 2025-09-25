

# %%
from typing import Optional
from _common import *
# 通过迭代后的策略模型进行自我对弈，生成对局数据
# 对于一局棋 获胜方所有的局面价值都为1 失败方所有的局面价值都为-1
# 忽略和棋的情况

import logging
import lightning as L
from torch.utils.data import TensorDataset, DataLoader

from chess.board import Board
from net.policy_net import PolicyNet
from net.value_net import ValueNet
from players.policy_player import PolicyPlayer
from utils import setup_logging, collect_selfplay_data, WORKSPACE
from utils.common import ValueCheckPointDir, split_dataset
from utils.db.loader import get_value_train_chess_records, get_value_train_data

# %%


class ValueTrain:
  checkpoint_dir: Path
  data_version: int
  model_version: int
  policy_model: PolicyNet
  train_record_limit: int
  max_epochs: int
  ###
  value_model: ValueNet
  trainer: L.Trainer | None

  def __init__(self, policy_model: PolicyNet, model_version: int, checkpoint_dir=ValueCheckPointDir, train_record_limit: int = 1000, max_epochs=100):
    self.model_version = model_version
    self.data_version = -abs(model_version)
    self.policy_model = policy_model
    self.checkpoint_dir = checkpoint_dir
    self.train_record_limit = train_record_limit
    self.max_epochs = max_epochs
    ###
    self.value_model = self._load_value_model()
    self.trainer = None

  @property
  def _model_name(self) -> str:
    return f"version_{self.model_version}.ckpt"

  def _load_value_model(self) -> ValueNet:
    # 加载已有的价值网络模型

    path = self.checkpoint_dir / f"{self._model_name}"
    if path.exists():
      logging.info(f"加载已有的价值网络模型 {path}")
      model = ValueNet.load_from_checkpoint(path)
      return model
    logging.info(f"未找到已有的价值网络模型 {path}，使用新模型")
    return ValueNet()

  def _gen_enough_data(self, train_record_limit: int = 1000, num_games: int = 100):
    # num_games 表示每次进行自我对弈的局数
    # train_record_limit 表示需要生成多少条有效的自我对弈数据后才进行训练
    # 使用模型生成足够的自我对弈数据
    while True:
      train_records = get_value_train_chess_records(version=self.data_version)
      if len(train_records) >= train_record_limit:
        logging.info(
            f"自我对弈数据已足够 当前数据版本:{self.data_version} 共有 {len(train_records)} 条")
        return
      logging.info(
          f"自我对弈数据不足 当前数据版本:{self.data_version} 仅有 {len(train_records)} 条 需要{train_record_limit} 条")

      red_player = PolicyPlayer("红方", model=self.policy_model, temperature=1.0)
      black_player = PolicyPlayer("黑方", model=self.policy_model, temperature=1.0)
      collect_selfplay_data(red_player, black_player, num_games=num_games,
                            version=self.data_version, draw_turns=200)

  def _train_model(self, max_epochs=100):
      # 加载最新的对弈数据
    states_tensor, values_tensor = get_value_train_data(
        version=self.data_version, chess_record_num=None)

    full_dataset = TensorDataset(states_tensor, values_tensor)
    train_dataset, val_dataset = split_dataset(full_dataset)

    logging.info(f"开始训练价值网络 共有 {len(train_dataset)} 条训练数据")
    model = self.value_model

    trainer = L.Trainer(max_epochs=max_epochs, default_root_dir=WORKSPACE)
    trainer.fit(model, train_dataloaders=DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0),
                val_dataloaders=DataLoader(val_dataset, batch_size=256,
                                           shuffle=False, num_workers=0),
                )
    self.trainer = trainer

  def _save_model(self):
    model_save_path = self.checkpoint_dir / f"version_{abs(self.data_version)}.ckpt"
    if not self.trainer:
      logging.warning("未进行训练, 无法保存模型")
      return
    logging.info(f"训练完成 模型保存路径:{model_save_path}")
    self.trainer.save_checkpoint(model_save_path)
    logging.info("保存模型完成")

  def run(self):
    logging.info(f"开始生成价值网络训练数据")
    self._gen_enough_data(train_record_limit=self.train_record_limit)
    logging.info(f"开始训练价值网络")
    self._train_model(max_epochs=self.max_epochs)
    self._save_model()

  def evaluate(self, board: Board) -> float:
    self.value_model.eval()
    state_tensor = board.to_network_input().unsqueeze(0)
    value = self.value_model(state_tensor)
    return value


# %%
if __name__ == "__main__":
  setup_logging()
  policy_version = 5
  model_version = 1
  policy_model = PolicyNet.load_from_checkpoint(
      PolicyCheckPointDir / f"version_{policy_version}.ckpt")
  # train = ValueTrain(policy_model=policy_model, model_version=model_version,
  #                    train_record_limit=2000, max_epochs=100)
  train = ValueTrain(policy_model=policy_model, model_version=model_version,
                     train_record_limit=3000, max_epochs=300)
  train.run()
# %%
