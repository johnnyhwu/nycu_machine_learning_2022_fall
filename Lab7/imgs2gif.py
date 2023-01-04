import os
from PIL import Image

PATH = "SNE/40"
START_IDX = 0
END_IDX = 500
STEP = 10

imgs = [Image.open(os.path.join(PATH, f"{i}.png")) for i in range(START_IDX, END_IDX, STEP)]

imgs[0].save(
    os.path.join(PATH, "result.gif"),
    save_all=True,
    append_images=imgs[1:],
    duration=150,
    loop=0
)

