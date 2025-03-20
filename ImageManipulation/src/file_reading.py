import numpy as np
import os
import os.path

FOLDERS = ["3DFilters", "DetailedFilters", "NormalFilters"]


def _type_verification(value):
    pass

def _read_3d_archive():
    pass

def _read_detailed_archives():
    pass

def _read_normal_archives():
    pass

def _find_file_folder(file_name: str):
    for folder in FOLDERS:
        if file_name in os.listdir(f"Filters/{folder}"):
            return folder

def read_file(file_name: str):
    match _find_file_folder(file_name):
        case "NormalFilters":
            pass

        case "DetailedFilters":
            pass

        case "3DFilters":
            pass

        case _:
            print(f"Filter {file_name} folder not found")
            return None