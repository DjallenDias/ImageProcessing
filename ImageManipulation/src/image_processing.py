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

def _img_to_rgb_arr(img: Image.Image):
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

def _g_in_rgb(img: Image.Image):
    g = img.getchannel("G")

    return Image.merge("RGB", (g, g, g))

def _yiq_y_in_rgb(img: Image.Image):
    r, g, b = _img_to_rgb_arr(img)

    y = 0.299*r + 0.587*g + 0.114*b
    y = y.astype(np.uint8)

    y_rgb = np.stack([y, y, y], axis=-1)

    return Image.fromarray(y_rgb, mode="RGB")

def colored_to_gray_img(img_name: str, mode: str = "RGB"):
    img = _open_image(img_name)

    if mode == "RGB":
        return _g_in_rgb(img)
    elif mode == "YIQ":
        return _yiq_y_in_rgb(img)
    else:
        print(f"Invalid mode {mode}.")
        return None