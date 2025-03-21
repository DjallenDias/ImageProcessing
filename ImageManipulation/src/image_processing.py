import numpy as np
import PIL.Image
from PIL import Image
import os.path as path

import src.file_reading as file_reading

def _open_image(img_name: str):
    img_path = f"img/{img_name}"
    return PIL.Image.open(img_path)

def _load_filter(filter_name: str):
    return file_reading.read_file(filter_name)

def _img_to_rgb_arr(img_name: str):
    img = _open_image(img_name)
    r = np.array(img.getchannel("R"))
    g = np.array(img.getchannel("G"))
    b = np.array(img.getchannel("B"))

    return r, g, b

def _rgb_to_img(r: np.ndarray, g: np.ndarray, b: np.ndarray):
    r_img = Image.fromarray(r)
    g_img = Image.fromarray(g)
    b_img = Image.fromarray(b)

    image_r = Image.merge("RGB", (r_img, Image.new("L", r_img.size, 0), Image.new("L", r_img.size, 0)))
    image_g = Image.merge("RGB", (Image.new("L", g_img.size, 0), g_img, Image.new("L", g_img.size, 0)))
    image_b = Image.merge("RGB", (Image.new("L", b_img.size, 0), Image.new("L", b_img.size, 0), b_img))

    return Image.merge("RGB", (r_img, g_img, b_img))