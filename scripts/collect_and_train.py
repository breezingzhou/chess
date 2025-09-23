# %%
from _common import *
# 加载模型自我对弈 生成对弈数据 并保存到数据库
# 从数据库加载对弈数据 训练策略模型
import logging
import lightning as L
import torch
from torch.utils.data import TensorDataset, DataLoader

from utils import show_images_in_slider, setup_logging, WORKSPACE
from utils.db import SelfPlayChessRecordDAL, SelfPlayChessRecord
from utils.db.loader import get_policy_train_data, get_selfplay_chess_records
from chess import ChessRecordData, Game
from net.policy_net import PolicyNet
from players.policy_player import PolicyPlayer

# %%

# %%


def test():
  from chess.utils import generate_board_images
  checkpoint_dir = WORKSPACE / "lightning_logs/version_17/checkpoints"
  checkpoint_path = list(checkpoint_dir.iterdir())[0]

  model = PolicyNet.load_from_checkpoint(checkpoint_path)
  red_player = PolicyPlayer("红方", model=model, temperature=2.0)
  black_player = PolicyPlayer("黑方", model=model, temperature=2.0)
  game = Game(red_player, black_player, debug=True, evaluate=True)
  result = game.start_play_loop()
  game.log_evaluation()
  movelist = game.movelist
  images = generate_board_images(movelist, show_last_pos=True)
  show_images_in_slider(images)


# %%
def collect_selfplay_data(red_player: PolicyPlayer, black_player: PolicyPlayer, version: int, num_games: int = 1000, draw_turns: int = 200, log_epoch: int = 10):
  # version 表示数据版本
  logging.info(f"开始自我对弈 对弈局数：{num_games}")
  for i in range(num_games):
    if i % log_epoch == 0:
      logging.info(f"开始第 {i + 1} / {num_games} 局对弈")
    game = Game(red_player, black_player, evaluate=True)

    winner = game.start_play_loop(draw_turns)
    r = SelfPlayChessRecord(
        id=None,
        red_player=red_player.display_name,
        black_player=black_player.display_name,
        winner=winner,
        movelist=game.movelist,
        version=version
    )
    SelfPlayChessRecordDAL.save_record(r)


# %%
CheckPointDir = WORKSPACE / "res/checkpoints/policy"
# base.ckpt 是使用大师对局数据训练的模型 epoch=46-step=49585
# 在train_policy.py中生成
# 手动复制到 CheckPointDir 下

# %%


class SelfPlayTrainLoop:
  checkpoint_dir: Path
  base_model: PolicyNet
  model_version: int

  def __init__(self, checkpoint_dir=CheckPointDir):
    self.checkpoint_dir = checkpoint_dir
    self.base_model = PolicyNet.load_from_checkpoint(checkpoint_dir / "base.ckpt")

    self.model_version = self._get_current_model_version()
    self.data_version = self.model_version + 1
    self.current_model = self._load_model(self.model_version)

  def _get_current_model_version(self) -> int:
    # 遍历 CheckPointDir 下的所有模型文件 获取最大的版本号
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
      collect_selfplay_data(red_player, black_player, num_games=100,
                            version=self.data_version, draw_turns=200, log_epoch=10)

  def _train_model(self, max_epochs=100):
      # 加载最新的对弈数据
    states, move_probs = get_policy_train_data(version=self.data_version, chess_record_num=None)
    states_tensor = torch.stack(states)
    move_probs_tensor = torch.stack(move_probs)
    train_dataset = TensorDataset(states_tensor, move_probs_tensor)
    # 训练模型
    logging.info(f"开始训练模型 当前数据版本:{self.data_version} 共有 {len(states)} 条训练数据")
    model = self.current_model
    trainer = L.Trainer(max_epochs=max_epochs)
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
  train = SelfPlayTrainLoop()
  train.selfplay_train_loop(turns=5, train_record_limit=500, max_epochs=100)

# %%
