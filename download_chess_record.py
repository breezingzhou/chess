# %%
# 从东萍象棋网下载棋谱
# 棋谱movelist格式为字符串 每4个字符表示一步走法
# 如“7747 7062” 表示第一步红方将棋子从77移动到47（炮二平五）  黑方将棋子从70移动到62（马8进7）
from pathlib import Path
from time import sleep
import httpx
from bs4 import BeautifulSoup
import re

# %%

# %%


def download_html(client: httpx.Client, url: str) -> str:
  response = client.get(url)
  response.raise_for_status()  # 确保请求成功
  return response.text


DhtmlXQ_movelist_var_name = "DhtmlXQ_movelist"
DHTMLXQ_MOVELIST_PATTERN = re.compile(
    rf'{re.escape(DhtmlXQ_movelist_var_name)}\s*=\s*(.*?);',
    re.DOTALL | re.IGNORECASE
)
MOVELIST_CLEANUP_PATTERN = re.compile(r'\[.*?\]')


def parse_movelist(html: str) -> str | None:
  soup = BeautifulSoup(html, 'html.parser')
  scripts = soup.find_all('script')

  for script in scripts:
    if script.text:
      match = DHTMLXQ_MOVELIST_PATTERN.search(script.text)
      if match:
        movelist_str = match.group(1)
        movelist_str = movelist_str.strip("'\"")
        movelist_str = MOVELIST_CLEANUP_PATTERN.sub('', movelist_str)
        return movelist_str
  return None

# %%


def init_httpx_client() -> httpx.Client:
  headers = {
      "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
  }
  client = httpx.Client(
      headers=headers,
      follow_redirects=True,
      verify=False
  )
  return client


# %%
save_file = Path(__file__).parent / "res/大师对局.dat"


def get_download_start_chess_number() -> int:
  if not save_file.exists():
    return 1
  else:
    with open(save_file, 'r', encoding='utf-8') as f:
      lines = f.readlines()
    if len(lines) == 0:
      return 1
    last_line = lines[-1].strip()
    return int(last_line.split(" ")[0]) + 1


def append_chess_records(records: list[tuple[int, str]]) -> None:
  print(f"正在保存对局记录...共{len(records)}局")
  with open(save_file, 'a', encoding='utf-8') as f:
    for record in records:
      f.write(f"{record[0]} {record[1]}\n")
# %%


def download_chess_records(records_num: int = 1000, save_epoch: int = 100) -> None:
  # 下载指定数量的对局记录
  # records_num: 下载的对局记录数量
  # save_epoch: 每多少局保存一次记录
  client = init_httpx_client()
  start_chess_no = get_download_start_chess_number()
  records = []
  for chess_no in range(start_chess_no, start_chess_no + records_num):
    sleep(0.5)  # 暂停 0.5 秒
    print(f"正在下载第 {chess_no} 局对局记录...")
    url = f"http://dpxq.com/hldcg/search/view_m_{chess_no}.html"
    try:
      html_content = download_html(client, url)
    except Exception as e:
      print(f"下载第 {chess_no} 局对局记录失败: {e}")
      continue

    movelist = parse_movelist(html_content)
    if movelist and len(movelist) % 4 == 0:
      records.append((chess_no, movelist))
    else:
      print(f"第 {chess_no} 局对局记录格式不正确 movelist_str: {movelist}")
      continue
    if len(records) >= save_epoch:
      append_chess_records(records)
      records = []  # 清空已保存的记录
  append_chess_records(records)


# %%
download_chess_records(records_num=200)

# %%
