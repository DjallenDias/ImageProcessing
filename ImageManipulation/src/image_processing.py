import numpy as np
import PIL.Image

import file_reading

def _open_image(img_name: str):
    return PIL.Image.open(img_name)

def _load_filter(filter_name: str):
    return file_reading.read_file(filter_name)

def _img_to_rgb_arr(img_name: str):
    img = _open_image(img_name)
    r = np.array(img.getchannel("R"))
    g = np.array(img.getchannel("G"))
    b = np.array(img.getchannel("B"))
    
    return r, g, b