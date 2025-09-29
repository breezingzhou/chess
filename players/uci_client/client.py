# %%
import asyncio
import concurrent.futures
import contextlib
import threading
from typing import Any, Callable, Coroutine, Generator, List, Optional, Union

from .protocol import UciProtocol
from .command import T, ConfigMapping
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


class UciClient:
  def __init__(self, transport: asyncio.SubprocessTransport, protocol: UciProtocol, *, timeout: Optional[float] = 10.0) -> None:
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

  def bestmove(self, fen: str) -> Optional[str]:
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
  def popen(cls, command: Union[str, List[str]], *, timeout: Optional[float] = 10.0, debug: Optional[bool] = None, **popen_args: Any) -> "UciClient":
    async def background(future: concurrent.futures.Future[UciClient]) -> None:
      transport, protocol = await UciProtocol.popen(command, **popen_args)
      threading.current_thread().name = f"{cls.__name__} (pid={transport.get_pid()})"
      client = cls(transport, protocol, timeout=timeout)
      try:
        await asyncio.wait_for(protocol.initialize(), timeout)
        future.set_result(client)
        returncode = await protocol.returncode
        client.returncode.set_result(returncode)
      finally:
        client.close()
      await client.shutdown_event.wait()

    return run_in_background(background, name=f"{cls.__name__} (command={command!r})", debug=debug)
