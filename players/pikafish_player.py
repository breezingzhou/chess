# %%
from pathlib import Path
import subprocess
import queue
import threading

from chess.board import Board
from chess.define import Move
from utils.common import WORKSPACE
from players.base_player import BasePlayer

# %%
PIKAFISH_EXE_PATH = WORKSPACE / "res/pikafish.exe"
# %%


class PikafishEngine:
  def __init__(self, engine_path):
    # 启动引擎进程，通过管道进行输入输出
    self.process = subprocess.Popen(
        engine_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # 以文本模式而非字节模式通信
        encoding="utf-8",
        universal_newlines=True
    )
    self.output_queue = queue.Queue()
    self.output_thread = threading.Thread(target=self._read_output)
    self.output_thread.daemon = True
    self.output_thread.start()
    _ = self.read_response()
    print("引擎已启动 pid=", self.process.pid)

  def _read_output(self):
    assert self.process.stdout is not None
    while True:
      line = self.process.stdout.readline()
      if not line:
        break
      self.output_queue.put(line.strip())

  def send_command(self, command):
    """向引擎发送命令"""
    if self.process.poll() is not None:
      raise RuntimeError("引擎进程已终止")

    assert self.process.stdin is not None
    self.process.stdin.write(command + "\n")  # 发送命令（末尾需加换行符）
    self.process.stdin.flush()  # 确保命令被立即发送
    print(f"发送命令: {command}")

  def read_response(self, timeout=0.2):
    """读取引擎的响应"""
    response: list[str] = []
    try:
      while True:
        line = self.output_queue.get(timeout=timeout)
        response.append(line)
    except queue.Empty:
      pass
    return response

  def __del__(self):
    self.close()

  def close(self):
    """关闭引擎进程"""
    if self.process.poll() is None:
      self.send_command("quit")
      try:
        self.process.wait(timeout=3)
      except Exception as e:
        print(f"关闭引擎时出错: {e}")
        self.process.kill()
    print("引擎已关闭")

# %%


class PikafishPlayer(BasePlayer):
  def __init__(self, name: str, engine_path: Path = PIKAFISH_EXE_PATH, evaluate: bool = False):
    super().__init__(name, evaluate)

    self.engine = PikafishEngine(engine_path)

  @property
  def display_name(self) -> str:
    return f"{self.name}_pikafish"

  def get_move(self, board: Board) -> Move:
    fen = board.to_fen()
    self.engine.send_command(f"position fen {fen}")
    movetime = 1000  # 设定搜索时间，单位为毫秒
    self.engine.send_command("go movetime {movetime}}")  # 搜索1秒
    timeout = 1.2 * movetime / 1000.0
    response = self.engine.read_response(timeout=timeout)
    last_line = response[-1] if response else ""
    bast_move_str = last_line.split()[0]
    bast_move = Move.from_uci_str(bast_move_str)
    return bast_move
