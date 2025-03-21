import numpy as np
import PIL.Image

def _open_image(file_name: str):
    return PIL.Image.open(file_name)