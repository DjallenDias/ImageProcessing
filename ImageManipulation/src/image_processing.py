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

def _crop_zeros(arr: np.ndarray):
    arr = arr.copy()
    arr = arr[~np.all(arr == 0, axis = 1)]
    arr = arr[:, ~np.all(arr == 0, axis = 0)]

    return arr

def _apply_filter_in_array(arr: np.ndarray, filter: np.ndarray,
                           offset: int = 0, step: int = 1, actv_func: str = ""):
    
    arr_res = np.zeros(arr.shape)

    for i in range(0, arr.shape[0], step):
        for j in range(0, arr.shape[1], step):

            arr_part = arr[i:i + filter.shape[0], j:j + filter.shape[1]]

            if arr_part.shape != filter.shape: continue

            arr_res[i,j] = np.sum(arr_part * filter)

    arr_res = _crop_zeros(arr_res)
    arr_res = arr_res + offset
    if actv_func:
        arr_res = np.vectorize(ReLU)(arr_res)

    return arr_res.astype(np.uint8)

def apply_filter(img_name: str, filter_name: str,
                 offset: int = 0, step: int = 1, actv_func: str = ""):
    
    img = _open_image(img_name)
    filter = _load_filter(filter_name)

    if type(filter) == dict:
        offset = filter["offset"]
        step = filter["step"]
        actv_func = filter["function"]
        filter = filter["filter"]

    elif type(filter) != np.ndarray:
        print(f"Invalid filter name {filter_name}")
        return None

    r, g, b = _img_to_rgb_arr(img)

    r = _apply_filter_in_array(r, filter, offset, step, actv_func)
    g = _apply_filter_in_array(g, filter, offset, step, actv_func)
    b = _apply_filter_in_array(b, filter, offset, step, actv_func)

    return _rgb_to_img(r, g, b)
