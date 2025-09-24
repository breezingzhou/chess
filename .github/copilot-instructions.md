# bz-chess AI Agent Instructions

简明指南：帮助 AI 代码助手在本仓库中快速高效做出正确修改。聚焦本项目特有结构、数据流、约定与开发工作流。

## 1. 项目定位 & 顶层结构
本项目实现中国象棋 (Xiangqi) 的 AlphaZero 风格训练 / 推理流水：
- `chess/` 棋盘与规则建模（`board.py` 负责局面、合法走子生成委托给 `rs_chinese_chess` Rust 引擎；`define.py` 定义核心数据结构与常量）。
- `utils/mcts.py` 新版 AlphaZero 风格 MCTS（策略 + 价值双网络）。
- 顶层旧版 PoC：根目录 `mcts.py`（单文件早期实现 + `MCTSPlayer`）。若行为重复，优先维护 `utils/mcts.py`。
- `net/` 策略 / 价值网络结构 (`policy_net.py`, `value_net.py`, `res_block.py`)；Lightning Checkpoints 写入 `lightning_logs/`。
- `players/` 不同落子策略封装（纯策略网络 `policy_player.py` 等）。
- `scripts/` 训练 / 自我对弈 / 数据收集与可视化脚本（命令行入口）。
- `res/` 数据与检查点持久化（对局记录、sqlite、本地图像资源、模型权重目录）。
- `chess_engine_rs/` Rust 辅助引擎（生成合法走子）。

## 2. 核心数据流（自我对弈 → 训练）
1. 生成局面：`Board` 初始化 / 执行 `do_move`。
2. 合法走子：`Board.available_moves()` 调用 Rust 封装 `rs_chinese_chess.Board.generate_move()`；返回 `Move` 列表（from/to 位置）。
3. 神经网络输入：`Board.to_network_input()` -> Tensor `[N_FEATURES, 10, 9]` (含当前执子方层 ±1)。
4. MCTS：`utils/mcts.run_mcts()` -> (selected_move, π(MOVE_SIZE), root)。π 来自根节点子节点访问计数归一化。
5. 训练数据（三元组）：(state tensor, π, z)。`z` 为终局赢家相对当前执子方的 ±1 / 和棋 0。
6. 训练：Lightning `PolicyNet` / `ValueNet`（详见各自 `load_from_checkpoint` 使用方式）。

## 3. 关键约定 / 特殊点
- 全局走子空间：`MOVE_SIZE`、`MOVE_TO_INDEX`、`LEGAL_MOVES` 在 `chess.define` 中（未展示文件时需先检索再改）。策略输出与 π 必须对齐此全局索引空间。
- 价值取值：始终是“当前执子方”视角 => 备份时符号翻转（`utils/mcts.MCTS._backup`）。
- 自我对弈探索：根节点加入 Dirichlet 噪声（α/ε 常量位于 `utils/mcts.py` 顶部，可通过参数覆盖）。
- 旧文件 `mcts.py` 中 `MCTSPlayer`、`MCTS` 和新版实现接口不同；新增功能请优先扩展新版，必要时写迁移适配层，而非双处复制逻辑。
- 图像渲染：`Board.to_image()` 使用 `PIL` + `res/simhei.ttf`；若新增可视化保持 API 幂等，不修改棋盘状态。
- 棋盘 FEN：`Board.to_fen_rs()` 与 `rs_chinese_chess` 交互的精简格式（仅 side to move）。`to_fen()` 额外拼接占位字段。
- 性能注意：`available_moves()` 内部重新构造 Rust Board；在循环内避免不必要重复深拷贝；MCTS 内部深拷贝已集中在入口或展开处。

## 4. 扩展 / 修改模式
- 添加新搜索策略：放入 `players/`，继承 `BasePlayer`，实现 `get_move(board)`；若复用 MCTS，调用 `utils.mcts.run_mcts`，不要再复制搜索逻辑。
- 新网络结构：置于 `net/`；保持 `.load_from_checkpoint()` 与现有 Lightning 兼容；必要时在 `common` 或配置处加入 checkpoint 目录常量。
- 新训练脚本：放入 `scripts/`，统一使用 `if __name__ == "__main__":` 保护，遵循现有脚本命名（`train_*.py`）。
- 修改 `Board` 行为时：同步评估是否影响 `to_network_input()` / MCTS 展开；任何改变合法走子集合的修改需回归测试生成的数据分布。
- 测试代码不要提供命令行版本，尽量使用函数式接口。
- 使用两个空格缩进。

## 5. 常用调用示例
策略落子：
```python
from chess.board import Board
from net.policy_net import PolicyNet
from players.policy_player import PolicyPlayer
b = Board()
model = PolicyNet.load_from_checkpoint('path/to.ckpt')
player = PolicyPlayer('p', model, temperature=0.8, device='cpu')
move = player.get_move(b)
b.do_move(move)
```
MCTS 一步：
```python
from utils.mcts import run_mcts
move, pi, root = run_mcts(b, policy_net=model, value_net=value_model, n_simulations=400, temperature=1.0)
```

## 6. 质量 / 审查重点
- 保持策略网络输出维度与 `MOVE_SIZE` 一致；新增层不改变最终线性头大小。
- MCTS 修改后验证：根节点 `children` 访问计数非全零；温度=0 时选择访问计数最大者。
- 不在热路径加入 I/O 或日志巨量输出（`search` 循环内慎用 logging）。
- 终局判定逻辑：若双方王都不存在 -> 和棋；任一缺失判胜；更复杂规则（如长将）尚未实现，不要假设存在。

## 7. 依赖 / 环境
- Python 3.13，依赖管理使用 `uv` (`pyproject.toml` + `uv.lock`)；Torch 通过自定义 index (CUDA 12.6)。
- 运行前确保虚拟环境激活；新增依赖写入 `pyproject.toml` `[project.dependencies]`。

## 8. 贡献建议 (面向 AI 修改)
- 先搜索是否存在新版实现：优先更新 `utils/mcts.py` 而不是根目录旧版。
- 引入新常量放 `chess.define` 或单独 `config.py`，避免硬编码散落。
- 若功能涉及持久化 / 棋局数据，复用 `res/` 目录结构；不要写入 `lightning_logs/` 之外的随机路径。
- 编写示例或 `_quick_test()` 风格本地函数时加 `# pragma: no cover` 以便未来集成测试。

## 9. 待补充（需要人工补资料）
- 训练脚本参数规范（`scripts/train_*.py` 现未分析，补充后可列出标准 CLI）。
- 数据收集与 replay 格式说明（`selfplay_chess_record_*.csv` 列含义）。

若你需要新增上述缺失部分，请标注 TODO 并最小侵入实现。

---
若本文件有缺口或描述不准确，可在 PR 中补充：集中在“数据流”“约定”章节。
