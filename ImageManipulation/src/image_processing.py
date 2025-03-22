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

def ReLU(value: int | float):
    return max(0, value)

def _crop_zeros(arr: np.ndarray):
    arr = arr.copy()
    arr = arr[~np.all(arr == 0, axis = 1)]
    arr = arr[:, ~np.all(arr == 0, axis = 0)]

    return arr

def _clip(arr: np.ndarray):
    arr = arr.clip(0, 255)
    return arr.astype(np.uint8)

def _wrap(arr: np.ndarray):
    arr = arr % 256
    return arr.astype(np.uint8)

def _abs(arr: np.ndarray):
    return np.abs(arr)

def _expansion_array(arr: np.ndarray):
    uniques = np.unique_counts(arr)
    values, _ = uniques
    aux_arr = np.zeros(values.shape)

    for i in range(values.shape[0]):
        aux = ((values[i] - values.min()) / (values.max() - values.min())) * 255
        aux_arr[i] = int(aux)

    map_d = dict(zip(values.flatten(), aux_arr.flatten()))

    vec_replace = np.vectorize(lambda x: map_d.get(x, x))

    res = np.array(vec_replace(arr), dtype=np.uint8)
    return res

def expansion(img_name: str):
    r, g, b = _img_to_rgb_arr(_open_image(img_name))
    r = _expansion_array(r)
    g = _expansion_array(g)
    b = _expansion_array(b)
    return _rgb_to_img(r, g, b)

def _equalization_array(arr: np.ndarray):
    uniques = np.unique_counts(arr)
    values, counts = uniques
    aux_arr = np.zeros(values.shape)
    sum = np.uint32(0)

    for i in range(values.shape[0]):
        sum += counts[i]
        aux_arr[i] = int((255/arr.size) * sum)

    map_d = dict(zip(values.flatten(), aux_arr.flatten()))

    vec_replace = np.vectorize(lambda x: map_d.get(x, x))

    res = np.array(vec_replace(arr), dtype=np.uint8)
    return res

def equalization(img_name: str):
    r, g, b = _img_to_rgb_arr(_open_image(img_name))
    r = _equalization_array(r)
    g = _equalization_array(g)
    b = _equalization_array(b)
    return _rgb_to_img(r, g, b)

def _overflow(arr: np.ndarray):
    return arr.max() > 255

def _underflow(arr: np.ndarray):
    return arr.min() < 0

def _crop_borders(arr: np.ndarray, filter: np.ndarray):
    center = (filter.shape[0], filter.shape[1])

    lines = arr.shape[0] - (center[0] - 1)
    cols = arr.shape[1] - (center[1] - 1)
    
    res_arr = arr[0:lines, 0:cols]
    
    return res_arr

def _apply_filter_in_array(arr: np.ndarray, filter: np.ndarray,
                           offset: int = 0, step: int = 1, actv_func: str = ""):
    
    arr_res = np.zeros(arr.shape)

    for i in range(0, arr.shape[0], step):
        for j in range(0, arr.shape[1], step):

            arr_part = arr[i:i + filter.shape[0], j:j + filter.shape[1]]

            if arr_part.shape != filter.shape: continue

            arr_res[i,j] = np.sum(arr_part * filter)

    if step > 1:
        arr_res = _crop_zeros(arr_res)

    else:
        arr_res = _crop_borders(arr_res, filter)
        pass

    arr_res = arr_res + offset

    if actv_func.lower() == "relu":
        arr_res = np.vectorize(ReLU)(arr_res)

    return arr_res.astype(np.int64)

def _apply_3d_filter_in_img(img: Image.Image, filter: np.ndarray,
                           offset: int = 0, step: int = 1, actv_func: str = ""):
    
    filter_r = filter[0]
    filter_g = filter[1]
    filter_b = filter[2]

    r, g, b = _img_to_rgb_arr(img)

    r = _apply_filter_in_array(r, filter_r, offset, step, actv_func)
    g = _apply_filter_in_array(g, filter_g, offset, step, actv_func)
    b = _apply_filter_in_array(b, filter_b, offset, step, actv_func)

    arr_res = r + g + b

    return arr_res

def apply_filter(img_name: str, filter_name: str,
                 offset: int = 0, step: int = 1, actv_func: str = "",
                 handle_overflow: str = "", handle_underflow: str = ""):
    
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
    
    if len(filter.shape) == 3:
        r = g = b = _apply_3d_filter_in_img(img, filter, offset, step, actv_func)
    
    else:
        r, g, b = _img_to_rgb_arr(img)

        r = _apply_filter_in_array(r, filter, offset, step, actv_func)
        g = _apply_filter_in_array(g, filter, offset, step, actv_func)
        b = _apply_filter_in_array(b, filter, offset, step, actv_func)

    handles_methods = {"clip": _clip,
                       "wrap": _wrap,
                       "expansion": _expansion_array,
                       "equalization": _equalization_array,
                       "absolute": _abs}

    if (_underflow(r) or _overflow(r) or
        _underflow(g) or _overflow(g) or
        _underflow(b) or _overflow(b)):

        if handle_overflow and handle_underflow:
            r = handles_methods[handle_overflow](r)
            g = handles_methods[handle_overflow](g)
            b = handles_methods[handle_overflow](b)

        else:
            print("RGB values greater than 255 or less than 0")
            return r, g, b

    r = r.astype(np.uint8)
    g = g.astype(np.uint8)
    b = b.astype(np.uint8)

    return _rgb_to_img(r, g, b)

def border_detection(img_name: str):
    img = _open_image(img_name)
    hor_sobel = _load_filter("hor_sobel.txt")
    ver_sobel = _load_filter("ver_sobel.txt")

    r, g, b = _img_to_rgb_arr(img)

    hr = _apply_filter_in_array(r, hor_sobel)
    hg = _apply_filter_in_array(g, hor_sobel)
    hb = _apply_filter_in_array(b, hor_sobel)

    vr = _apply_filter_in_array(r, ver_sobel)
    vg = _apply_filter_in_array(g, ver_sobel)
    vb = _apply_filter_in_array(b, ver_sobel)

    r = np.abs(hr) + np.abs(vr)
    g = np.abs(hg) + np.abs(vg)
    b = np.abs(hb) + np.abs(vb)

    r = _expansion_array(r)
    g = _expansion_array(g)
    b = _expansion_array(b)

    return _rgb_to_img(r, g, b)