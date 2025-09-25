"""AlphaGo 风格的蒙特卡洛树搜索 (MCTS with Rollout)

与 utils/mcts.py (AlphaZero 风格) 的区别：
- 叶节点评估阶段引入快速 rollout（随机或自定义策略）直至终局/步数上限，得到 rollout 价值。
- 将价值网络评估值与 rollout 价值按权重线性混合，提高早期对局面评估的稳定性。

核心流程：
1) 选择 Selection: 使用 PUCT 公式从根递归选择子节点。
2) 扩展 Expansion: 使用策略网络为合法走子分配先验概率，并生成子节点。
3) 评估 Evaluation: 终局直接返回价值；否则混合 价值网络评估 与 Rollout 结果。
4) 回传 Backup: 沿路径回传统计 (N, W)，每层价值取负以切换视角。
5) 输出 Policy: 根节点子节点访问计数归一化为 π，温度控制探索。

注意：
- 价值恒以“当前执子方”视角衡量；备份时层层取负。
- 策略网络输出维度需与全局 MOVE_SIZE 对齐；π 同样基于 MOVE_TO_INDEX 映射。

使用示例：
  from chess.board import Board
  from net.policy_net import PolicyNet
  from net.value_net import ValueNet
  from utils.mcts_go import run_mcts_go

  board = Board()
  policy = PolicyNet.load_from_checkpoint('path/to/policy.ckpt').eval().to('cpu')
  value = ValueNet.load_from_checkpoint('path/to/value.ckpt').eval().to('cpu')
  move, pi, root = run_mcts_go(board, policy, value, n_simulations=400, temperature=1.0)
  print('建议走子:', move)
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import copy
import math
import random

import torch
import torch.nn.functional as F

from chess.board import Board
from chess.define import Move, MOVE_TO_INDEX, MOVE_SIZE, ChessWinner
from net.policy_net import PolicyNet
from net.value_net import ValueNet


# -------------------------------------------------------------
# 可调参数（可在构造/调用时覆盖）
# -------------------------------------------------------------
DEFAULT_C_PUCT = 3
DEFAULT_DIRICHLET_ALPHA = 0.3
DEFAULT_DIRICHLET_EPS = 0.25

# AlphaGo 风格：rollout 设置
DEFAULT_ROLLOUT_WEIGHT = 0.5      # 混合权重: v = (1-w)*v_net + w*v_rollout
DEFAULT_ROLLOUT_MAX_STEPS = 200   # rollout 最大步数（避免超长对局）


@dataclass
class MCTSGoNode:
  board: Board
  parent: Optional["MCTSGoNode"] = None
  prior: float = 0.0
  move: Optional[Move] = None

  visits: int = 0
  value_sum: float = 0.0

  children: Dict[Move, "MCTSGoNode"] = field(default_factory=dict)
  # 在节点上缓存先验，但不一次性创建所有子节点，实现逐步扩展
  priors: Dict[Move, float] = field(default_factory=dict)
  expanded: bool = False
  terminal: bool = False
  terminal_value: float = 0.0

  def q_value(self) -> float:
    return 0.0 if self.visits == 0 else self.value_sum / self.visits

  def u_value(self, c_puct: float) -> float:
    if self.parent is None:
      return 0.0
    return c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)

  def get_value(self, c_puct: float) -> float:
    return self.q_value() + self.u_value(c_puct)

  def select(self, c_puct: float) -> Tuple[Move, "MCTSGoNode"]:
    assert self.children, "选择失败：节点未展开或无子节点"
    return max(self.children.items(), key=lambda kv: kv[1].get_value(c_puct))

  def child_stats(self) -> List[Tuple[Move, "MCTSGoNode"]]:
    return list(self.children.items())

  def unexpanded_moves(self) -> List[Move]:
    if not self.expanded:
      return []
    return [mv for mv in self.priors.keys() if mv not in self.children]


class MCTSGo:
  def __init__(
      self,
      policy_net: PolicyNet,
      value_net: ValueNet,
      *,
      c_puct: float = DEFAULT_C_PUCT,
      dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
      dirichlet_epsilon: float = DEFAULT_DIRICHLET_EPS,
      rollout_weight: float = DEFAULT_ROLLOUT_WEIGHT,
      rollout_max_steps: int = DEFAULT_ROLLOUT_MAX_STEPS,
      rollout_policy: Optional[Callable[[Board, List[Move]], Move]] = None,
      device: str | torch.device = "cuda",
  ) -> None:
    self.policy_net = policy_net.eval()
    self.value_net = value_net.eval()
    self.c_puct = c_puct
    self.dirichlet_alpha = dirichlet_alpha
    self.dirichlet_epsilon = dirichlet_epsilon
    self.rollout_weight = rollout_weight
    self.rollout_max_steps = rollout_max_steps
    self.rollout_policy = rollout_policy
    self.device = device

  # -------------------------- 主流程 -------------------------
  def search(self, board: Board, n_simulations: int, add_dirichlet: bool = True) -> MCTSGoNode:
    root_board = copy.deepcopy(board)
    root = MCTSGoNode(board=root_board)
    self._expand(root)  # 仅计算先验，不创建子节点

    # 根节点在有先验时加入 Dirichlet 噪声（无需已创建子节点）
    if add_dirichlet and not root.terminal:
      self._add_dirichlet_noise(root)

    for _ in range(n_simulations):
      leaf, path = self._select(root)
      value = self._evaluate_and_expand(leaf)
      self._backup(path, value)

    return root

  def get_policy(self, root: MCTSGoNode, temperature: float = 1.0) -> torch.Tensor:
    visit_counts = torch.zeros(MOVE_SIZE, dtype=torch.float32)
    if root.terminal:
      return visit_counts
    moves_nodes = root.child_stats()
    if temperature <= 1e-6:
      best_move, _ = max(moves_nodes, key=lambda kv: kv[1].visits)
      visit_counts[MOVE_TO_INDEX[best_move]] = 1.0
      return visit_counts

    counts = torch.tensor([node.visits for _, node in moves_nodes], dtype=torch.float32)
    counts = counts ** (1.0 / max(temperature, 1e-6))
    if counts.sum() > 0:
      probs = counts / counts.sum()
      for (mv, _), p in zip(moves_nodes, probs.tolist()):
        visit_counts[MOVE_TO_INDEX[mv]] = p
    return visit_counts

  def select_move(self, root: MCTSGoNode, temperature: float = 1.0) -> Move:
    assert not root.terminal, "终局无法选择走子"
    moves_nodes = root.child_stats()
    if temperature <= 1e-6:
      return max(moves_nodes, key=lambda kv: kv[1].visits)[0]
    counts = torch.tensor([child.visits for _, child in moves_nodes], dtype=torch.float32)
    counts = counts ** (1.0 / max(temperature, 1e-6))
    probs = counts / counts.sum()
    idx = int(torch.multinomial(probs, 1).item())
    return moves_nodes[idx][0]

  # -------------------------- 内部细节 -----------------------
  def _add_dirichlet_noise(self, root: MCTSGoNode) -> None:
    # 对根节点全部先验添加噪声（即使尚未实例化子节点）
    if not root.expanded or root.terminal or not root.priors:
      return
    moves = list(root.priors.keys())
    n = len(moves)
    if n == 0:
      return
    noise = torch.distributions.Dirichlet(torch.full((n,), self.dirichlet_alpha)).sample().tolist()
    for mv, nval in zip(moves, noise):
      root.priors[mv] = root.priors[mv] * (1 - self.dirichlet_epsilon) + nval * self.dirichlet_epsilon
    # 同步已创建子节点的 prior
    for mv, child in root.children.items():
      if mv in root.priors:
        child.prior = root.priors[mv]

  def _select(self, root: MCTSGoNode) -> Tuple[MCTSGoNode, List[MCTSGoNode]]:
    path: List[MCTSGoNode] = [root]
    node = root
    while True:
      # 终局或未扩展节点：在 evaluate 阶段处理
      if node.terminal:
        return node, path

      if not node.expanded:
        # 首次到达该节点：计算先验，并仅扩展一个子节点
        self._expand(node)
        if node.terminal:
          return node, path
        # 随机选择一个未扩展动作创建子节点
        mv = random.choice(list(node.priors.keys())) if node.priors else None
        if mv is None:
          return node, path
        child = self._create_child(node, mv)
        path.append(child)
        return child, path

      # 已扩展节点：若存在未扩展动作，则随机扩展一个并终止选择；否则沿最佳子节点向下
      unexp = node.unexpanded_moves()
      if unexp:
        mv = random.choice(unexp)
        child = self._create_child(node, mv)
        path.append(child)
        return child, path

      # 没有未扩展动作，选择已存在子节点中 PUCT 最高者继续向下
      if not node.children:
        return node, path
      next_move, next_child = max(node.children.items(), key=lambda kv: kv[1].get_value(self.c_puct))
      node = next_child
      path.append(node)

  def _expand(self, node: MCTSGoNode) -> None:
    if node.expanded:
      return

    is_end, winner = node.board.game_end()
    if is_end:
      node.terminal = True
      node.expanded = True
      if winner == ChessWinner.Draw:
        node.terminal_value = 0.0
      else:
        # 胜利者与当前执子方一致则 +1，否则 -1
        node.terminal_value = 1.0 if winner.number == node.board.current_turn.number else -1.0
      return

    legal_moves = node.board.available_moves()
    if not legal_moves:
      node.terminal = True
      node.expanded = True
      node.terminal_value = 0.0
      return

    policy_probs = self._policy_eval(node)
    ids = [MOVE_TO_INDEX[mv] for mv in legal_moves]
    probs = policy_probs[ids]
    probs_sum = float(probs.sum().item())
    if probs_sum <= 0:
      probs = torch.full((len(legal_moves),), 1.0 / len(legal_moves), dtype=torch.float32)
    else:
      probs = probs / probs.sum()

    node.priors = {mv: float(p) for mv, p in zip(legal_moves, probs.tolist())}
    node.expanded = True

  def _create_child(self, node: MCTSGoNode, mv: Move) -> MCTSGoNode:
    new_board = copy.deepcopy(node.board)
    new_board.do_move(mv)
    prior = node.priors.get(mv, 0.0)
    child = MCTSGoNode(board=new_board, parent=node, prior=float(prior), move=mv)
    node.children[mv] = child
    return child

  def _evaluate_and_expand(self, node: MCTSGoNode) -> float:
    if node.terminal:
      return node.terminal_value

    # 按“每次只扩展一个未探索节点”原则：到达的新子节点只做评估，不再继续扩展
    # 如果节点未标记 expanded（新创建的叶子），直接评估其价值
    # 若是早先存在但从未扩展的节点（极少路径下可能发生），也只评估一次

    # 价值网络评估
    v_net = self._value_eval(node)
    # rollout 评估
    v_roll = self._rollout_value(node)
    # 加权混合
    value = (1.0 - self.rollout_weight) * v_net + self.rollout_weight * v_roll
    return float(value)

  def _backup(self, path: List[MCTSGoNode], leaf_value: float) -> None:
    value = leaf_value
    for node in reversed(path):
      node.visits += 1
      node.value_sum += value
      value = -value

  def _policy_eval(self, node: MCTSGoNode) -> torch.Tensor:
    state = node.board.to_network_input().unsqueeze(0).to(self.device)
    with torch.no_grad():
      logits = self.policy_net(state)  # [1, MOVE_SIZE]
    probs = F.softmax(logits, dim=-1).squeeze(0).cpu()
    return probs

  def _value_eval(self, node: MCTSGoNode) -> float:
    state = node.board.to_network_input().unsqueeze(0).to(self.device)
    with torch.no_grad():
      v = self.value_net(state).item()
    return float(v)

  # -------------------------- Rollout ------------------------
  def _rollout_value(self, node: MCTSGoNode) -> float:
    start_turn = node.board.current_turn.number
    bd = copy.deepcopy(node.board)

    for _ in range(self.rollout_max_steps):
      is_end, winner = bd.game_end()
      if is_end:
        if winner == ChessWinner.Draw:
          return 0.0
        return 1.0 if winner.number == start_turn else -1.0

      legal = bd.available_moves()
      if not legal:
        # 无棋可走视为负
        return -1.0

      mv = self._sample_rollout_move(bd, legal)
      bd.do_move(mv)

    # 步数上限未分胜负，判和
    return 0.0

  def _sample_rollout_move(self, board: Board, legal_moves: List[Move]) -> Move:
    if self.rollout_policy is not None:
      try:
        return self.rollout_policy(board, legal_moves)
      except Exception:
        # 回退到随机策略，避免外部策略异常导致崩溃
        pass
    # 默认：均匀随机
    return random.choice(legal_moves)


# -------------------------------------------------------------
# 便捷函数
# -------------------------------------------------------------
def run_mcts_go(
    board: Board,
    policy_net: PolicyNet,
    value_net: ValueNet,
    *,
    n_simulations: int = 800,
    temperature: float = 1.0,
    add_dirichlet: bool = True,
    **kwargs,
) -> Tuple[Move, torch.Tensor, MCTSGoNode]:
  """执行 AlphaGo 风格的 MCTS 搜索。

  返回 (selected_move, policy_tensor, root_node)
    - selected_move: 根据 temperature 从根访问计数选择/采样得到的走子
    - policy_tensor: 大小为 (MOVE_SIZE,) 的策略分布（根子节点访问计数归一化）
    - root_node: 根节点（包含整棵搜索树，便于调试/可视化）
  """
  mcts = MCTSGo(policy_net=policy_net, value_net=value_net, **kwargs)
  root = mcts.search(board, n_simulations=n_simulations, add_dirichlet=add_dirichlet)
  pi = mcts.get_policy(root, temperature=temperature)
  mv = mcts.select_move(root, temperature=temperature)
  return mv, pi, root
