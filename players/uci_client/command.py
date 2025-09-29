# %%
from __future__ import annotations

import asyncio
import enum
import logging
from typing import Callable, Generic, List, Mapping, Type, TypeVar, Union, override

# %%
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from .protocol import Protocol, UciProtocol


T = TypeVar("T")
LOGGER = logging.getLogger(__name__)

ConfigValue = Union[str, int, bool, None]
ConfigMapping = Mapping[str, ConfigValue]
# %%


class EngineError(RuntimeError):
  pass


class CommandTimeoutError(EngineError):
  pass


class EngineTerminatedError(EngineError):
  pass
# %%


class CommandState(enum.Enum):
  NEW = enum.auto()
  ACTIVE = enum.auto()
  # CANCELLING = enum.auto()
  DONE = enum.auto()
# %%


class BaseCommand(Generic[T]):
  def __init__(self, engine: "Protocol", timeout: float = 10.0) -> None:
    self.engine = engine

    self.result: asyncio.Future[T] = asyncio.Future()
    self.timeout = timeout

    self.state = CommandState.NEW

  def _engine_terminated(self, code: int) -> None:
    exc = EngineTerminatedError(f"engine process died unexpectedly (exit code: {code})")
    self._handle_exception(exc)

  def _handle_exception(self, exc: Exception) -> None:
    if not self.result.done():
      self.result.set_exception(exc)

  def _start(self) -> None:
    assert self.state == CommandState.NEW, self.state
    self.state = CommandState.ACTIVE
    try:
      self.check_initialized()
      self.start()
    except EngineError as err:
      self._handle_exception(err)

  def _line_received(self, line: str) -> None:
    assert self.state in [CommandState.ACTIVE], self.state
    try:
      self.line_received(line)
    except EngineError as err:
      self._handle_exception(err)

  def cancel(self) -> None:
    pass

  def start(self) -> None:
    raise NotImplementedError

  def check_initialized(self) -> None:
    if not self.engine.initialized:
      raise EngineError("tried to run command, but engine is not initialized")

  def line_received(self, line: str) -> None:
    pass


# %%
def _next_token(line: str) -> tuple[str, str]:

  parts = line.split(maxsplit=1)
  return parts[0] if parts else "", parts[1] if len(parts) == 2 else ""


# %%


class UciInitializeCommand(BaseCommand[None]):
  def __init__(self, engine: "UciProtocol"):
    super().__init__(engine)

  @override
  def check_initialized(self) -> None:
    pass

  @override
  def start(self) -> None:
    self.engine.send_line("uci")

  @override
  def line_received(self, line: str) -> None:
    token, remaining = _next_token(line)
    if token == "uciok":
      self.result.set_result(None)


class UciQuitCommand(BaseCommand[None]):
  def __init__(self, engine: "UciProtocol"):
    super().__init__(engine)

  @override
  def start(self) -> None:
    self.engine.send_line("quit")


class UciOptionsCommand(BaseCommand[None]):
  def __init__(self, engine: "UciProtocol", options: ConfigMapping):
    super().__init__(engine)
    self.options = options

  @override
  def start(self) -> None:
    for name, value in self.options.items():
      if value:
        command = f"setoption name {name} value {value}"
      else:
        command = f"setoption name {name}"
      self.engine.send_line(command)
    self.result.set_result(None)


class UciPositionCommand(BaseCommand[None]):
  def __init__(self, engine: "UciProtocol", fen: str):
    super().__init__(engine)
    self.fen = fen

  @override
  def start(self) -> None:
    self.engine.send_line(f"position fen {self.fen}")
    self.result.set_result(None)


class UciGoCommand(BaseCommand[str]):
  def __init__(self, engine: "UciProtocol", depth: int):
    super().__init__(engine)
    self.depth = depth

  @override
  def start(self) -> None:
    self.engine.send_line(f"go depth {self.depth}")

  @override
  def line_received(self, line: str) -> None:
    token, remaining = _next_token(line)
    if token == "bestmove":
      bestmove, _ = _next_token(remaining)
      self.result.set_result(bestmove)


# %%
