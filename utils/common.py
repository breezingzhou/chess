# %%
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent

CheckPointDir = WORKSPACE / "res/checkpoints"
PolicyCheckPointDir = CheckPointDir / "policy"
# base.ckpt 是使用大师对局数据训练的模型 epoch=46-step=49585
# 在train_policy.py中生成
# 手动复制到 PolicyCheckPointDir 下
