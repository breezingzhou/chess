"""AlphaZero 风格蒙特卡洛树搜索(MCTS)

核心思想:
1. 选择 (Selection): 从根节点开始, 依据 PUCT 公式递归挑选子节点直到到达叶节点。
2. 扩展 (Expansion): 对叶节点调用策略网络得到所有合法走子的先验概率, 为每个合法走子创建子节点。
3. 评估 (Evaluation): 价值网络评估当前局面的价值 v (范围[-1,1] 表示当前执子方视角)。终局直接以胜负结果映射为价值。
4. 回传 (Backup): 沿搜索路径反向更新每个节点 (N,W,Q), 并在每一层交替取负值 (因为对手的好即自己的坏)。
5. 输出 (Policy): 使用根节点子节点访问计数 N 归一化得到改进策略 π, 温度 τ 控制探索 (τ=0 取访问计数最大者)。

注: 本实现保持与 AlphaZero 论文概念一致, 但针对中国象棋使用全局 MOVE 列表+索引的策略网络输出格式。

依赖:
  - `Board.available_moves()` 返回当前合法 `Move` 列表
  - `Board.do_move(move)` 改变棋盘 (内部会轮转 `current_turn`)
  - `Board.game_end()` 判断终局并给出胜者
  - `board.to_network_input()` -> Tensor shape (8,H,W)
  - 策略网络输出 shape (B, MOVE_SIZE) 未 softmax 的 logits
  - 价值网络输出 shape (B,1) 经过 tanh 已在 [-1,1]

训练数据生成: 自我对弈时可记录 (state, π, z) 其中 π 为根节点访问计数归一化后映射到全局 MOVE_SIZE 的向量, z 为最终胜负结果从该 state 当前执子方视角的价值。

使用示例:
  from chess.board import Board
  from net.policy_net import PolicyNet
  from net.value_net import ValueNet
  from utils.mcts import run_mcts

  board = Board()
  policy_net = PolicyNet().to('cpu')
  value_net = ValueNet().to('cpu')
  move, pi, root = run_mcts(board, policy_net, value_net, n_simulations=200, temperature=1.0)
  print('下一步推荐:', move, '策略分布非零数:', int((pi>0).sum()))
"""
# %%
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import math

import torch
import torch.nn.functional as F

from chess.board import Board
from chess.define import Move, MOVE_TO_INDEX, MOVE_SIZE, ChessWinner
from net.policy_net import PolicyNet
from net.value_net import ValueNet


# -------------------------------------------------------------
# 配置参数 (可在使用时传入覆盖)
# -------------------------------------------------------------
DEFAULT_C_PUCT = 1.4          # PUCT 探索常数
DEFAULT_DIRICHLET_ALPHA = 0.3  # Dirichlet 噪声 alpha (针对合法走子集合)
DEFAULT_DIRICHLET_EPS = 0.25  # 根节点混合系数: (1-eps)*P + eps*Dir


@dataclass
class MCTSNode:
  board: Board                       # 当前局面 (该节点代表的状态)
  parent: Optional['MCTSNode'] = None
  prior: float = 0.0                 # P(s,a) 先验概率 (来自策略网络)
  move: Optional[Move] = None        # 从父节点到达该节点的动作 (根节点为 None)

  # 统计量
  visits: int = 0                    # N(s,a)
  value_sum: float = 0.0             # W(s,a)

  # 展开信息
  children: Dict[Move, 'MCTSNode'] = field(default_factory=dict)
  is_expanded: bool = False
  terminal: bool = False             # 是否终局
  terminal_value: float = 0.0        # 若终局, 该局面价值(当前执子方视角)

  @property
  def q_value(self) -> float:
    return 0.0 if self.visits == 0 else self.value_sum / self.visits

  @property
  def expanded(self) -> bool:
    return self.is_expanded

  def value_for_backup(self) -> float:
    if self.terminal:
      return self.terminal_value
    return self.q_value

  def child_stats(self) -> List[Tuple[Move, 'MCTSNode']]:
    return list(self.children.items())


class MCTS:
  def __init__(
          self,
          policy_net: PolicyNet,
          value_net: ValueNet,
          c_puct: float = DEFAULT_C_PUCT,
          dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
          dirichlet_epsilon: float = DEFAULT_DIRICHLET_EPS,
          device: str | torch.device = 'cpu'):
    self.policy_net = policy_net.eval()  # 推理阶段 eval
    self.value_net = value_net.eval()
    self.c_puct = c_puct
    self.dirichlet_alpha = dirichlet_alpha
    self.dirichlet_epsilon = dirichlet_epsilon
    self.device = device

  # -------------------------- 对外接口 -------------------------
  def search(self, board: Board, n_simulations: int, add_dirichlet: bool = True) -> MCTSNode:
    """对给定局面执行若干次模拟, 返回根节点。"""
    # 深拷贝根局面, 保证外部不被修改
    import copy
    root_board = copy.deepcopy(board)
    root = MCTSNode(board=root_board)
    self._expand(root)

    if add_dirichlet and not root.terminal and len(root.children) > 0:
      self._add_dirichlet_noise(root)

    for _ in range(n_simulations):
      node, path = self._select(root)
      value = self._evaluate_and_expand(node)
      self._backup(path, value)

    return root

  def get_policy(self, root: MCTSNode, temperature: float = 1.0) -> torch.Tensor:
    """返回基于访问计数的策略向量 π (MOVE_SIZE 大小)"""
    visit_counts = torch.zeros(MOVE_SIZE, dtype=torch.float32)
    if root.terminal:
      return visit_counts  # 没有后继
    moves_nodes = root.child_stats()
    if temperature <= 1e-6:  # 近似 0 温度: 选最大访问计数
      best_move, _ = max(moves_nodes, key=lambda kv: kv[1].visits)
      visit_counts[MOVE_TO_INDEX[best_move]] = 1.0
      return visit_counts

    # τ > 0: N^{1/τ}
    counts = torch.tensor([child.visits for _, child in moves_nodes], dtype=torch.float32)
    counts = counts ** (1.0 / temperature)
    if counts.sum() > 0:
      probs = counts / counts.sum()
      for (move, _), p in zip(moves_nodes, probs.tolist()):
        visit_counts[MOVE_TO_INDEX[move]] = p
    return visit_counts

  def select_move(self, root: MCTSNode, temperature: float = 1.0) -> Move:
    """按照温度从根节点访问计数采样/选择一个走子。"""
    assert not root.terminal, "终局无法继续走子"
    moves_nodes = root.child_stats()
    if temperature <= 1e-6:
      return max(moves_nodes, key=lambda kv: kv[1].visits)[0]
    counts = torch.tensor([child.visits for _, child in moves_nodes], dtype=torch.float32)
    counts = counts ** (1.0 / temperature)
    probs = counts / counts.sum()
    idx_tensor = torch.multinomial(probs, num_samples=1)
    idx: int = int(idx_tensor.item())
    return moves_nodes[idx][0]

  # -------------------------- 内部流程 -------------------------
  def _add_dirichlet_noise(self, root: MCTSNode):
    legal_children = list(root.children.values())
    n = len(legal_children)
    noise = torch.distributions.Dirichlet(torch.full((n,), self.dirichlet_alpha)).sample().tolist()
    for (child, n_val) in zip(legal_children, noise):
      child.prior = child.prior * (1 - self.dirichlet_epsilon) + n_val * self.dirichlet_epsilon

  def _select(self, root: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
    """向下选择直到到达叶节点或终局。返回叶节点与路径(含叶)。"""
    path: List[MCTSNode] = [root]
    node = root
    while node.expanded and not node.terminal:
      if not node.children:
        break
      total_visits = sum(child.visits for child in node.children.values())
      best_score = -float('inf')
      best_child: Optional[MCTSNode] = None
      for move, child in node.children.items():
        ucb = self._puct_score(parent=node, child=child, total_visits=total_visits)
        if ucb > best_score:
          best_score = ucb
          best_child = child
      assert best_child is not None
      node = best_child
      path.append(node)
      if node.terminal:
        break
    return node, path

  def _puct_score(self, parent: MCTSNode, child: MCTSNode, total_visits: int) -> float:
    # PUCT: Q + c*P*sqrt(sumN)/(1+N)
    q = child.q_value
    prior = child.prior
    u = self.c_puct * prior * math.sqrt(max(1, total_visits)) / (1 + child.visits)
    return q + u

  def _expand(self, node: MCTSNode):
    if node.is_expanded:
      return
    # 终局判断
    is_end, winner = node.board.game_end()
    if is_end:
      node.terminal = True
      if winner == ChessWinner.Draw:
        node.terminal_value = 0.0
      else:
        # winner.number: Red=1, Black=-1
        # 当前执子方视角价值: 若 winner.color == current_turn => +1 否则 -1
        # winner.number 与当前 turn.number 的乘积即可
        current_turn_number = node.board.current_turn.number
        node.terminal_value = 1.0 if winner.number == current_turn_number else -1.0
      node.is_expanded = True
      return

    legal_moves = node.board.available_moves()
    if not legal_moves:
      # 无合法走子视为和棋
      node.terminal = True
      node.terminal_value = 0.0
      node.is_expanded = True
      return

    # 使用策略网络获取先验
    state = node.board.to_network_input().unsqueeze(0).to(self.device)  # [1,8,H,W]
    with torch.no_grad():
      policy_logits = self.policy_net(state)  # [1,MOVE_SIZE]
    policy_probs = F.softmax(policy_logits, dim=-1).squeeze(0).cpu()  # [MOVE_SIZE]

    # TODO check policy_probs -> legal_moves_policy_probs
    for mv in legal_moves:
      idx = MOVE_TO_INDEX[mv]
      prior = float(policy_probs[idx].item())
      # 创建子局面
      import copy
      new_board = copy.deepcopy(node.board)
      new_board.do_move(mv)
      child = MCTSNode(board=new_board, parent=node, prior=prior, move=mv)
      node.children[mv] = child
    node.is_expanded = True

  def _evaluate_and_expand(self, node: MCTSNode) -> float:
    # 如果已是终局 或 已扩展 (含终局), 返回价值
    if node.terminal:
      return node.terminal_value
    if not node.expanded:
      # 还未展开, 先展开(包含策略)
      self._expand(node)
      if node.terminal:  # 展开发现终局
        return node.terminal_value

    # 使用价值网络评估当前节点局面 视角=当前执子方
    state = node.board.to_network_input().unsqueeze(0).to(self.device)
    with torch.no_grad():
      value = self.value_net(state).item()  # 已经在 [-1,1]
    return float(value)

  def _backup(self, path: List[MCTSNode], leaf_value: float):
    # leaf_value: 最后节点价值 (其局面当前执子方视角)
    # 回传: 沿路径倒序, 节点价值符号交替
    value = leaf_value
    for node in reversed(path):
      node.visits += 1
      node.value_sum += value
      value = -value  # 视角转换


# -------------------------------------------------------------
# 便捷函数
# -------------------------------------------------------------
def run_mcts(
        board: Board,
        policy_net: PolicyNet,
        value_net: ValueNet,
        n_simulations: int = 800,
        temperature: float = 1.0,
        add_dirichlet: bool = True,
        **kwargs) -> Tuple[Move, torch.Tensor, MCTSNode]:
  """执行一次 MCTS 搜索并给出推荐走子与策略向量。

  返回: (selected_move, policy_tensor, root_node)
    - selected_move: 根据 temperature 选出的走子
    - policy_tensor: 大小 (MOVE_SIZE,) 的策略概率 (根子节点访问计数归一化)
    - root_node: 根节点 (可用来调试)
  """
  mcts = MCTS(policy_net=policy_net, value_net=value_net, **kwargs)
  root = mcts.search(board, n_simulations=n_simulations, add_dirichlet=add_dirichlet)
  policy = mcts.get_policy(root, temperature=temperature)
  move = mcts.select_move(root, temperature=temperature)
  return move, policy, root
