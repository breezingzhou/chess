# %%
# 从东萍象棋网下载棋谱
# 棋谱movelist格式为字符串 每4个字符表示一步走法
# 如“7747 7062” 表示第一步红方将棋子从77移动到47（炮二平五）  黑方将棋子从70移动到62（马8进7）
from pathlib import Path
from time import sleep
import httpx
from bs4 import BeautifulSoup, Tag
import re
from dataclasses import dataclass
import polars as pl
# %%


@dataclass
class ChessRecord:
  red_player: str
  black_player: str
  type: str | None
  gametype: str | None
  result: str
  movelist: str
  chess_no: int = 0


# %%
def gen_pattern(tag: str) -> re.Pattern:
  return re.compile(
      rf"\[{re.escape(tag)}\](.*?)\[\/{re.escape(tag)}\]",
      re.DOTALL | re.IGNORECASE
  )


def download_html(client: httpx.Client, url: str) -> str:
  response = client.get(url)
  response.raise_for_status()  # 确保请求成功
  return response.text


DhtmlXQ_movelist_var_name = "DhtmlXQ_movelist"
MOVELIST_PATTERN = re.compile(
    rf'{re.escape(DhtmlXQ_movelist_var_name)}\s*=\s*(.*?);',
    re.DOTALL | re.IGNORECASE
)
MOVELIST_CLEANUP_PATTERN = re.compile(r'\[.*?\]')

REDPLAYER_PATTERN = gen_pattern("DhtmlXQ_redname")
BLACKPLAYER_PATTERN = gen_pattern("DhtmlXQ_blackname")
RESULT_PATTERN = gen_pattern("DhtmlXQ_result")
TYPE_PATTERN = gen_pattern("DhtmlXQ_type")
GAMETYPE_PATTERN = gen_pattern("DhtmlXQ_gametype")


def extract_content(text: str, pattern: re.Pattern) -> str | None:
  match = pattern.search(text)
  if match:
    return match.group(1).strip()
  return None


def parse_html(html: str) -> ChessRecord | None:
  soup = BeautifulSoup(html, 'html.parser')
  dhtmlxq_view = soup.find(id='dhtmlxq_view')
  scripts = soup.find_all('script')
  if not dhtmlxq_view or len(scripts) == 0:
    return None
  red_player = extract_content(dhtmlxq_view.text, REDPLAYER_PATTERN)
  black_player = extract_content(dhtmlxq_view.text, BLACKPLAYER_PATTERN)
  result = extract_content(dhtmlxq_view.text, RESULT_PATTERN)
  type_ = extract_content(dhtmlxq_view.text, TYPE_PATTERN)
  gametype = extract_content(dhtmlxq_view.text, GAMETYPE_PATTERN)
  movelist = parse_movelist(scripts)
  if red_player and black_player and movelist and result:
    return ChessRecord(
        red_player=red_player,
        black_player=black_player,
        type=type_,
        gametype=gametype,
        movelist=movelist,
        result=result
    )
  return None


def parse_movelist(scripts: Tag) -> str | None:
  for script in scripts:
    if script.text:
      match = MOVELIST_PATTERN.search(script.text)
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


def get_download_start_chess_number(output_file: Path) -> int:
  if not output_file.exists():
    return 1
  df = pl.read_csv(output_file, schema_overrides={"movelist": pl.String})
  return df['chess_no'].max() + 1


def save_chess_records(records: list[ChessRecord], output_file: Path) -> None:
  print(f"正在保存对局记录...共{len(records)}局")
  if len(records) == 0:
    return
  df = pl.DataFrame(records)
  cols = df.columns
  cols.insert(0, cols.pop(cols.index("chess_no")))  # chess_no放到首位
  df_reordered = df.select(cols)
  if not output_file.exists():
    df_reordered.write_csv(output_file)
  else:
    df = pl.read_csv(output_file, schema_overrides={"movelist": pl.String})
    df_union = pl.concat([df, df_reordered])
    df_union = df_union.unique(subset=["chess_no"])  # 去重
    df_union.sort("chess_no")
    df_union.write_csv(output_file)
# %%


def download_chess_records(output_file: Path, start_chess_no: int | None = None, records_num: int = 1000, save_epoch: int = 500) -> None:
  # 下载指定数量的对局记录
  # records_num: 下载的对局记录数量
  # save_epoch: 每多少局保存一次记录
  client = init_httpx_client()
  if start_chess_no is None:
    start_chess_no = get_download_start_chess_number(output_file)
  records: list[ChessRecord] = []
  for chess_no in range(start_chess_no, start_chess_no + records_num):
    sleep(1)
    print(f"正在下载第 {chess_no} 局对局记录...")
    url = f"http://dpxq.com/hldcg/search/view_m_{chess_no}.html"
    try:
      html_content = download_html(client, url)
    except Exception as e:
      print(f"下载第 {chess_no} 局对局记录失败: {e}")
      continue

    chessgame = parse_html(html_content)
    if chessgame and len(chessgame.movelist) % 4 == 0:
      chessgame.chess_no = chess_no
      records.append(chessgame)
    else:
      print(f"第 {chess_no} 局对局记录解析失败")
      continue
    if len(records) >= save_epoch:
      save_chess_records(records, output_file)
      records = []  # 清空已保存的记录
  save_chess_records(records, output_file)


# %%
# 截至2025.9.17共有 135423 局
output_file = Path(__file__).parent.parent / "res/大师对局.csv"
records_num = 135423
download_chess_records(output_file=output_file, records_num=records_num, save_epoch=500)


# %%
# client = init_httpx_client()
# chess_no = 29
# url = f"http://dpxq.com/hldcg/search/view_m_{chess_no}.html"
# html_content = download_html(client, url)
# chessgame = parse_html(html_content)

# %%
