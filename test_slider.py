# %%
from pathlib import Path
from ipywidgets import interact, IntSlider
from PIL import Image
from IPython.display import display


# %%
image_dir = Path("C:\\Users\\breezing\\Pictures\\caiqing\\zhuzi")
images = []
for img_path in image_dir.glob("*.jpg"):
  img = Image.open(img_path)
  images.append(img)

# %%
# %%


def show_images_in_slider(images):
  """在滑块中显示所有图片"""
  slider = IntSlider(
      min=0,
      max=len(images) - 1,
      step=1,
      value=0,
      description='图片索引:'
  )

  # display_handle = None

  # def show_image(index):
  #   """显示指定索引的图片"""
  #   nonlocal display_handle
  #   if display_handle is None:
  #     display_handle = display(images[index], display_id=True)
  #   else:
  #     display_handle.update(images[index])

  def show_image(index):
    display(images[index])
  interact(show_image, index=slider)


show_images_in_slider(images)

# %%
