# %%
import abc
import concurrent.futures
import contextlib
from pathlib import Path
import threading
from typing import Any, Callable, Coroutine, Generator, List, Optional, Self, Type, TypeVar, Union, override
import asyncio
import logging

from chess.engine import BaseCommand, CommandState, _next_token, ConfigMapping


# %%


T = TypeVar("T")

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
    self.next_command: Optional[BaseCommand[Any]] = None

    self.initialized = False
    self.returncode: asyncio.Future[int] = asyncio.Future()

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
    command, self.command = self.command, None
    next_command, self.next_command = self.next_command, None
    if command:
      command._engine_terminated(code)
    if next_command:
      next_command._engine_terminated(code)

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

  async def communicate(self, command_factory: Callable[[Self], BaseCommand[T]]) -> T:
    command = command_factory(self)

    if self.returncode.done():
      raise Exception(f"engine process dead (exit code: {self.returncode.result()})")

    assert command.state == CommandState.NEW, command.state

    if self.next_command is not None:
      self.next_command.result.cancel()
      self.next_command.finished.cancel()
      self.next_command.set_finished()

    self.next_command = command

    def previous_command_finished() -> None:
      self.command, self.next_command = self.next_command, None
      if self.command is not None:
        cmd = self.command

        def cancel_if_cancelled(result: asyncio.Future[T]) -> None:
          if result.cancelled():
            cmd._cancel()

        cmd.result.add_done_callback(cancel_if_cancelled)
        cmd._start()
        cmd.add_finished_callback(previous_command_finished)

    if self.command is None:
      previous_command_finished()
    elif not self.command.result.done():
      self.command.result.cancel()
    elif not self.command.result.cancelled():
      self.command._cancel()

    return await command.result

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

# %%


def run_in_background(coroutine: Callable[[concurrent.futures.Future[T]], Coroutine[Any, Any, None]], *, name: Optional[str] = None, debug: Optional[bool] = None) -> T:
  """
  Runs ``coroutine(future)`` in a new event loop on a background thread.

  Blocks on *future* and returns the result as soon as it is resolved.
  The coroutine and all remaining tasks continue running in the background
  until complete.
  """
  assert asyncio.iscoroutinefunction(coroutine)

  future: concurrent.futures.Future[T] = concurrent.futures.Future()

  def background() -> None:
    try:
      asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
      asyncio.run(coroutine(future), debug=debug)
      future.cancel()
    except Exception as exc:
      future.set_exception(exc)

  threading.Thread(target=background, name=name).start()
  return future.result()

# %%


class UciProtocol(Protocol):
  def __init__(self) -> None:
    super().__init__()

  async def initialize(self) -> None:
    class UciInitializeCommand(BaseCommand[None]):
      def __init__(self, engine: UciProtocol):
        super().__init__(engine)  # type: ignore[call-arg]
        self.engine = engine

      @override
      def check_initialized(self) -> None:
        if self.engine.initialized:
          raise Exception("engine already initialized")

      @override
      def start(self) -> None:
        self.engine.send_line("uci")

      @override
      def line_received(self, line: str) -> None:
        token, remaining = _next_token(line)
        if line.strip() == "uciok" and not self.result.done():
          self.engine.initialized = True
          self.result.set_result(None)
          self.set_finished()
    return await self.communicate(UciInitializeCommand)

  async def go(self) -> str:
    class UciGoCommand(BaseCommand[str]):
      def __init__(self, engine: UciProtocol):
        super().__init__(engine)  # type: ignore[call-arg]
        self.engine = engine

      @override
      def start(self) -> None:
        self.engine.send_line("go depth 20")

      @override
      def line_received(self, line: str) -> None:
        token, remaining = _next_token(line)
        if token == "info":
          pass
          # self._info(remaining)
        elif token == "bestmove":
          bestmove, _ = _next_token(remaining)
          self.result.set_result(bestmove)
    return await self.communicate(UciGoCommand)

  async def position(self, fen: str) -> None:
    self.send_line(f"position fen {fen}")

  async def quit(self) -> None:
    self.send_line("quit")

  async def configure(self, options: ConfigMapping) -> None:
    for k, v in options.items():
      if v is None:
        self.send_line(f"setoption name {k}")
      else:
        self.send_line(f"setoption name {k} value {v}")

  async def bestmove(self, fen: str) -> str:
    await self.position(fen)
    return await self.go()

# %%


class SimpleEngine:
  def __init__(self, transport: asyncio.SubprocessTransport, protocol: Protocol, *, timeout: Optional[float] = 10.0) -> None:
    self.transport = transport
    self.protocol = protocol
    self.timeout = timeout

    self._shutdown_lock = threading.Lock()
    self._shutdown = False
    self.shutdown_event = asyncio.Event()

    self.returncode: concurrent.futures.Future[int] = concurrent.futures.Future()

  @contextlib.contextmanager
  def _not_shut_down(self) -> Generator[None, None, None]:
    with self._shutdown_lock:
      if self._shutdown:
        raise Exception("engine event loop dead")
      yield

  def quit(self) -> None:
    with self._not_shut_down():
      coro = asyncio.wait_for(self.protocol.quit(), self.timeout)
      future = asyncio.run_coroutine_threadsafe(coro, self.protocol.loop)
    return future.result()

  def bestmove(self, fen: str) -> str:
    with self._not_shut_down():
      coro = asyncio.wait_for(self.protocol.bestmove(fen), self.timeout)
      future = asyncio.run_coroutine_threadsafe(coro, self.protocol.loop)
    return future.result()

  def configure(self, options: ConfigMapping) -> None:
    with self._not_shut_down():
      coro = asyncio.wait_for(self.protocol.configure(options), self.timeout)
      future = asyncio.run_coroutine_threadsafe(coro, self.protocol.loop)
    return future.result()

  def close(self) -> None:
    """
    Closes the transport and the background event loop as soon as possible.
    """
    def _shutdown() -> None:
      self.transport.close()
      self.shutdown_event.set()

    with self._shutdown_lock:
      if not self._shutdown:
        self._shutdown = True
        self.protocol.loop.call_soon_threadsafe(_shutdown)

  def __del__(self):
    self.quit()

  @classmethod
  def popen(cls, Protocol: Type[Protocol], command: Union[str, List[str]], *, timeout: Optional[float] = 10.0, debug: Optional[bool] = None, **popen_args: Any) -> "SimpleEngine":
    async def background(future: concurrent.futures.Future[SimpleEngine]) -> None:
      transport, protocol = await Protocol.popen(command, **popen_args)
      threading.current_thread().name = f"{cls.__name__} (pid={transport.get_pid()})"
      simple_engine = cls(transport, protocol, timeout=timeout)
      try:
        await asyncio.wait_for(protocol.initialize(), timeout)
        future.set_result(simple_engine)
        returncode = await protocol.returncode
        simple_engine.returncode.set_result(returncode)
      finally:
        simple_engine.close()
      await simple_engine.shutdown_event.wait()

    return run_in_background(background, name=f"{cls.__name__} (command={command!r})", debug=debug)


# %%
from utils import WORKSPACE


def test():
  pikapath = WORKSPACE / "res/pikafish.exe"
  nnuepath = WORKSPACE / "res/pikafish.nnue"
  engine = SimpleEngine.popen(UciProtocol, str(pikapath))
  options = {"EvalFile": str(nnuepath)}
  engine.configure(options=options)
  fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABN1 w - - 0 1"
  bestmove = engine.bestmove(fen)
  print("Best move:", bestmove)
  engine.quit()
# %%


def get_pikafish_engine():
  pikapath = WORKSPACE / "res/pikafish.exe"
  engine = SimpleEngine.popen(UciProtocol, str(pikapath))

  return engine
# %%
