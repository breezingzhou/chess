
# %%
import abc
import concurrent.futures
import contextlib
from pathlib import Path
import threading
from typing import Any, Callable, Coroutine, Generator, List, Mapping, Optional, Self, Type, TypeVar, Union, override
import asyncio
import logging
from .command import BaseCommand, T, CommandTimeoutError, ConfigMapping, EngineError, EngineTerminatedError

# %%
LOGGER = logging.getLogger(__name__)


# %%


class Protocol(asyncio.SubprocessProtocol):
  """Protocol for communicating with a chess engine process."""

  returncode: asyncio.Future[int]
  """Future: Exit code of the process."""

  def __init__(self) -> None:
    self.loop = asyncio.get_running_loop()
    self.transport: Optional[asyncio.SubprocessTransport] = None

    self.buffer = {
        1: bytearray(),  # stdout
        2: bytearray(),  # stderr
    }

    self.command: Optional[BaseCommand[Any]] = None
    # 防止多个并发 communicate 竞争启动命令
    self._command_lock = asyncio.Lock()

    self.initialized = False
    # 运行时设置进程退出码 future
    self.returncode = asyncio.Future()

  def connection_made(self, transport: asyncio.BaseTransport) -> None:
    # SubprocessTransport expected, but not checked to allow duck typing.
    self.transport = transport  # type: ignore
    LOGGER.debug("%s: Connection made", self)

  def connection_lost(self, exc: Optional[Exception]) -> None:
    assert self.transport is not None
    code = self.transport.get_returncode()
    assert code is not None, "connect lost, but got no returncode"
    LOGGER.debug("%s: Connection lost (exit code: %d, error: %s)", self, code, exc)

    # Terminate commands.

    if self.command:
      if isinstance(self.command, UciQuitCommand):
        self.command.result.set_result(None)
      else:
        self.command._engine_terminated(code)
    self.returncode.set_result(code)

  def process_exited(self) -> None:
    LOGGER.debug("%s: Process exited", self)

  def send_line(self, line: str) -> None:
    LOGGER.debug("%s: << %s", self, line)
    assert self.transport is not None, "cannot send line before connection is made"
    stdin = self.transport.get_pipe_transport(0)
    # WriteTransport expected, but not checked to allow duck typing.
    stdin.write((line + "\n").encode("utf-8"))  # type: ignore

  def pipe_data_received(self, fd: int, data: Union[bytes, str]) -> None:
    self.buffer[fd].extend(data)  # type: ignore
    while b"\n" in self.buffer[fd]:
      line_bytes, self.buffer[fd] = self.buffer[fd].split(b"\n", 1)
      if line_bytes.endswith(b"\r"):
        line_bytes = line_bytes[:-1]
      try:
        line = line_bytes.decode("utf-8")
      except UnicodeDecodeError as err:
        LOGGER.warning("%s: >> %r (%s)", self, bytes(line_bytes), err)
      else:
        if fd == 1:
          self._line_received(line)
        else:
          self.error_line_received(line)

  def error_line_received(self, line: str) -> None:
    LOGGER.warning("%s: stderr >> %s", self, line)

  def _line_received(self, line: str) -> None:
    LOGGER.debug("%s: >> %s", self, line)

    self.line_received(line)

    if self.command:
      self.command._line_received(line)

  def line_received(self, line: str) -> None:
    pass

  async def communicate(self, command: BaseCommand[T]) -> Optional[T]:
    if self.returncode.done():
      raise EngineTerminatedError("engine already terminated")

    result = None
    async with self._command_lock:
      self.command = command
      command._start()

      try:
        result = await asyncio.wait_for(command.result, timeout=command.timeout)
      except asyncio.TimeoutError:
        command._handle_exception(CommandTimeoutError(f"command timeout after {command.timeout}s"))
      finally:
        self.command = None
    return result

  def __repr__(self) -> str:
    pid = self.transport.get_pid() if self.transport is not None else "?"
    return f"<{type(self).__name__} (pid={pid})>"

  @abc.abstractmethod
  async def initialize(self) -> None:
    pass

  @abc.abstractmethod
  async def quit(self) -> None:
    pass

  @abc.abstractmethod
  async def configure(self, options: ConfigMapping) -> None:
    pass

  @abc.abstractmethod
  async def bestmove(self, fen: str) -> str:
    pass

  @classmethod
  async def popen(cls, command: Union[str, List[str]], **popen_args: Any):
    if not isinstance(command, list):
      command = [command]
    loop = asyncio.get_running_loop()
    return await loop.subprocess_exec(cls, *command, **popen_args)


# %%
from .command import UciInitializeCommand, UciPositionCommand, UciQuitCommand, UciGoCommand, UciOptionsCommand


class UciProtocol(Protocol):
  """UCI protocol implementation."""

  async def initialize(self) -> None:
    await self.communicate(UciInitializeCommand(self))
    self.initialized = True

  async def quit(self) -> None:
    command = UciQuitCommand(self)
    await self.communicate(command)

  async def configure(self, options: ConfigMapping) -> None:
    command = UciOptionsCommand(self, options)
    await self.communicate(command)

  async def bestmove(self, fen: str) -> Optional[str]:
    pos_command = UciPositionCommand(self, fen)
    await self.communicate(pos_command)
    go_command = UciGoCommand(self, depth=15)
    return await self.communicate(go_command)
# %%
