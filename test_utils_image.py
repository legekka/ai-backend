from modules.utils import *
from PIL import Image

test_image = Image.open("test/test2.jpg").convert("RGB")

process_image(test_image)