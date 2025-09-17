# %%
from chess.utils import generate_board_images, get_chess_records

from utils.show_in_slider import show_images_in_slider


def test_replay_chess_record():
  chess_records = get_chess_records()
  movelist_str = chess_records[1]
  images = generate_board_images(movelist_str)
  show_images_in_slider(images)


# %%
test_replay_chess_record()
# %%
