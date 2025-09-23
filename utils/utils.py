# %%
import time
from functools import wraps


def timer(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed
  return wrapper


# %%
import logging
import os
from datetime import datetime
from .common import WORKSPACE


def setup_logging(level=logging.INFO) -> None:
  log_dir = WORKSPACE / "logs"
  log_dir.mkdir(parents=True, exist_ok=True)
  log_filename = f"{log_dir}/{datetime.now().strftime('%Y%m%d')}.log"

  logging.basicConfig(
      level=level,  # 日志级别：DEBUG < INFO < WARNING < ERROR < CRITICAL
      format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
      datefmt='%Y-%m-%d %H:%M:%S',  # 时间格式
      handlers=[
          logging.FileHandler(log_filename, encoding='utf-8'),  # 写入文件
          logging.StreamHandler()  # 输出到控制台
      ]
  )
