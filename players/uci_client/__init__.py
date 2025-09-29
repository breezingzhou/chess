from .protocol import UciProtocol
from .client import UciClient
from utils import WORKSPACE


def get_pikafish_client():
  pika_exe_path = WORKSPACE / "res/pikafish.exe"
  client = UciClient.popen(str(pika_exe_path))
  return client
