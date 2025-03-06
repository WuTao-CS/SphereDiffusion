import torch
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import os


def create_color_dict():
    return {(0, 0, 0): 0,
            (174, 199, 232): 1,
            (152, 223, 138): 2,
            (31, 119, 180): 3,
            (255, 187, 120): 4,
            (188, 189, 34): 5,
            (140, 86, 75): 6,
            (255, 152, 150): 7,
            (214, 39, 40): 8,
            (197, 176, 213): 9,
            (148, 103, 189): 10,
            (196, 156, 148): 11,
            (23, 190, 207): 12,
            (178, 76, 76): 13,
            (247, 182, 210): 14,
            (66, 188, 102): 15,
            (219, 219, 141): 16,
            (140, 57, 197): 17,
            (202, 185, 52): 18,
            (51, 176, 203): 19,
            (200, 54, 131): 20,
            (92, 193, 61): 21,
            (78, 71, 183): 22,
            (172, 114, 82): 23,
            (255, 127, 14): 24,
            (91, 163, 138): 25,
            (153, 98, 156): 26,
            (140, 153, 101): 27,
            (158, 218, 229): 28,
            (100, 125, 154): 29,
            (178, 127, 135): 30,
            (120, 185, 128): 31,
            (146, 111, 194): 32,
            (44, 160, 44): 33,
            (112, 128, 144): 34,
            (96, 207, 209): 35,
            (227, 119, 194): 36,
            (213, 92, 176): 37,
            (94, 106, 211): 38,
            (82, 84, 163): 39,
            (100, 85, 144): 40}


with open('datasets/Structured3D/all_data_label_final.json', 'rt') as f:
    all_data = json.load(f)


colorDict = create_color_dict()
root = 'datasets/Structured3D'


def RGBtoOneHot(rgb, colorDict):
    arr = np.zeros((rgb.shape[0], rgb.shape[1], 1), dtype=np.uint8)
    for label, color in enumerate(colorDict.keys()):
        color = np.array(color)
        if label < len(colorDict.keys()):
            arr[np.all(rgb == color, axis=-1)] = label
    return arr


for item in tqdm(all_data):
    source_filename = item['source']
    try:
        source = Image.open(os.path.join(root, source_filename)).convert('RGB')
    except IOError:
        print(os.path.join(root, source_filename))
    source = np.array(source)
    source = RGBtoOneHot(source, colorDict)
    mask_image = Image.fromarray(source.squeeze(), mode='L')
    save_path = os.path.join(root, os.path.split(source_filename)
                             [0], 'semantic_label.png')
    mask_image.save(save_path)
