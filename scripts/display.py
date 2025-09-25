# %%
from matplotlib import pyplot as plt
import torch
from _common import *

# 统计各种对局指标
from chess.board import Board
from chess.define import ChessColor, ChessWinner
from chess.utils import generate_board_images, parse_movelist_str
from net.value_net import ValueNet
from utils import show_images_in_slider
from utils.common import ValueCheckPointDir
from utils.db import SelfPlayChessRecordDAL, SelfPlayChessRecordModel

# %%


def show_stats():
  for version in range(1, 5 + 1):
    records = SelfPlayChessRecordDAL.query(filters=[SelfPlayChessRecordModel.version == version])
    total_games = len(records)
    red_win_records = [r for r in records if r.winner == ChessWinner.Red.number]
    draw_win_records = [r for r in records if r.winner == ChessWinner.Draw.number]
    print(f"数据版本 {version} 对局总数: {total_games}")
    print(f"红方胜局数: {len(red_win_records)} 胜率: {len(red_win_records) / total_games:.2%}")
    print(f"和局数: {len(draw_win_records)} 和率: {len(draw_win_records) / total_games:.2%}")
# %%


def display_board_images(movelist: str):
  images = generate_board_images(movelist, show_last_pos=True)
  show_images_in_slider(images)

# %%


def evaluate_values(movelist: str, value_model: ValueNet) -> list[float]:
  """返回每一步走子前的价值(初始局面索引0)。"""
  moves = parse_movelist_str(movelist)
  b = Board()
  values: list[float] = []
  for i in range(len(moves) + 1):  # 包含初始局面和最后走子后的终局局面前价值
    state = b.to_network_input().unsqueeze(0)
    with torch.no_grad():
      v = value_model(state).item()  # 当前执子方视角
    values.append(float(v))
    if i == len(moves):
      break
    # 执行下一步
    b.do_move(moves[i])
  return values


def plot_curve(values: list[float], dpi: int = 100):
  plt.figure(figsize=(10, 4), dpi=dpi)
  plt.plot(range(len(values)), values, marker='o', linewidth=1.2)
  plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
  plt.xlabel('Play (半回合)')
  plt.ylabel('Value')
  plt.grid(alpha=0.3)
  plt.show()
  plt.close()


# %%

def test():
  records = SelfPlayChessRecordDAL.query(filters=[SelfPlayChessRecordModel.version == -1])
  movelist = records[0].movelist
  value_model = ValueNet.load_from_checkpoint(ValueCheckPointDir / "version_1.ckpt")
  values = evaluate_values(movelist, value_model)
  plot_curve(values, dpi=100)
  # images = generate_board_images(movelist, show_last_pos=True)
  # for image, value in zip(images, values):
  #   plt.imshow(image)
  #   plt.title(f"Value: {value:.3f}")
  #   plt.axis('off')
  #   plt.show()
  #   plt.close()


# %%
test()
# %%
