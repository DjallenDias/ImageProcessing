import numpy as np
import os
import os.path
from io import TextIOWrapper

FOLDERS = ["3DFilters", "DetailedFilters", "NormalFilters"]


def _type_verification(value):
    pass

def _read_3d_files(file: TextIOWrapper):
    pass

def _read_detailed_files(file: TextIOWrapper):
    pass

def _read_normal_files(file: TextIOWrapper):
    values_list = []
    readlines = file.readlines()
    file.close()

    for line in readlines:
        line_itens = []
        for item in line.replace(",", ".").replace("\n", "").split():
            item_ver = _type_verification(item)

            if item_ver != None:
                line_itens.append(item_ver)
        
        values_list.append(line_itens)

    return np.array(values_list)

def _find_file_folder(file_name: str):
    for folder in FOLDERS:
        if file_name in os.listdir(f"Filters/{folder}"):
            return folder

def read_file(file_name: str):
    match _find_file_folder(file_name):
        case "NormalFilters":
            return _read_normal_files(open(f"Filters/NormalFilters/{file_name}"))

        case "DetailedFilters":
            pass

        case "3DFilters":
            pass

        case _:
            print(f"Filter {file_name} folder not found")
            return None