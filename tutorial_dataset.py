import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from annotator.canny import CannyDetector
import random
from annotator.util import HWC3
from einops import repeat
from numba import jit
from annotator.equirect_rotation_fast import Rot_Equirect


def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out


def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    lst = [X % shape[1], Y % shape[0]]
    out = np.concatenate(lst, axis=-1)
    out_int = np.concatenate(lst, axis=-1)

    return out, out_int


def create_color_palette():
    return [
        (0, 0, 0),              # background
        (174, 199, 232),		# wall
        (152, 223, 138),		# floor
        (31, 119, 180), 		# cabinet
        (255, 187, 120),		# bed
        (188, 189, 34), 		# chair
        (140, 86, 75),  		# sofa
        (255, 152, 150),		# table
        (214, 39, 40),  		# door
        (197, 176, 213),		# window
        (148, 103, 189),		# bookshelf
        (196, 156, 148),		# picture
        (23, 190, 207), 		# counter
        (178, 76, 76),
        (247, 182, 210),		# desk
        (66, 188, 102),
        (219, 219, 141),		# curtain
        (140, 57, 197),
        (202, 185, 52),
        (51, 176, 203),
        (200, 54, 131),
        (92, 193, 61),
        (78, 71, 183),
        (172, 114, 82),
        (255, 127, 14), 		# refrigerator
        (91, 163, 138),
        (153, 98, 156),
        (140, 153, 101),
        (158, 218, 229),		# shower curtain
        (100, 125, 154),
        (178, 127, 135),
        (120, 185, 128),
        (146, 111, 194),
        (44, 160, 44),  		# toilet
        (112, 128, 144),		# sink
        (96, 207, 209),
        (227, 119, 194),		# bathtub
        (213, 92, 176),
        (94, 106, 211),
        (82, 84, 163),  		# otherfurn
        (100, 85, 144)       # otherprop
    ]


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


class Equirectangular:
    def __init__(self, img_name):
        # self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        self._img = Image.open(img_name).convert('RGB')
        self._img = np.array(self._img)
        self._height, self._width, _ = self._img.shape
        self._mask_img = np.zeros((self._height, self._width, 3), np.uint8)
        self._mask = np.ones((self._height, self._width))

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        # print(self._img.shape)
        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0,  1],
        ], np.float32)
        K_inv = np.linalg.inv(K)

        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate(
            [x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        xyz = xyz @ R.T
        lonlat = xyz2lonlat(xyz)
        XY, XY_INT = lonlat2XY(lonlat, shape=(self._height, self._width, 3))
        XY = XY.astype(np.float32)
        XY_INT = XY_INT.astype(np.int32)
        persp = cv2.remap(
            self._img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        # 遍历XY，将对应位置的Mask置为1
        for i in range(XY.shape[0]):
            for j in range(XY.shape[1]):
                # self._mask_img[int(XY[i][j][1]%self._height)][int(XY[i][j][0]%self._width)] = self._img[int(XY[i][j][1]%self._height)][int(XY[i][j][0]%self._width)]
                # self._mask[int(XY[i][j][1]%self._height)][int(XY[i][j][0]%self._width)] = 0
                self._mask_img[XY_INT[i][j][1]][XY_INT[i][j][0]
                                                ] = self._img[XY_INT[i][j][1]][XY_INT[i][j][0]]
                self._mask[XY_INT[i][j][1]][XY_INT[i][j][0]] = 0
        return persp, self._mask_img, self._mask


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        self.root = 'datasets/Structured3D'
        with open('datasets/Structured3D/all_data_label_final.json', 'rt') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        try:
            source = Image.open(os.path.join(
                self.root, source_filename)).convert('RGB')
        except IOError:
            print(os.path.join(self.root, source_filename))
        try:
            target = Image.open(os.path.join(
                self.root, target_filename)).convert('RGB')
        except IOError:
            print(os.path.join(self.root, target_filename))

        source = np.array(source)
        target = np.array(target)
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class CannyDataset(Dataset):
    def __init__(self):
        self.data = []
        self.root = 'datasets/Structured3D'
        self.apply_canny = CannyDetector()
        with open('datasets/Structured3D/all_data_label_train.json', 'rt') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item['target']
        target_filename = item['target']
        prompt = item['prompt']
        try:
            target = Image.open(os.path.join(
                self.root, target_filename)).convert('RGB')
        except IOError:
            print(os.path.join(self.root, target_filename))
        target = np.array(target)

        low_threshold = random.randint(1, 255)
        high_threshold = random.randint(1, 255)
        if low_threshold > high_threshold:
            t = low_threshold
            low_threshold = high_threshold
            high_threshold = t
        source = self.apply_canny(target, low_threshold, high_threshold)
        source = HWC3(source)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class Perspect2PanoDataset(Dataset):
    def __init__(self):
        self.data = []
        self.root = 'datasets/Structured3D'
        with open('datasets/Structured3D/new_all_data_label_train.json', 'rt') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item['target']
        target_filename = item['target']
        prompt = item['prompt']
        try:
            target = Image.open(os.path.join(
                self.root, target_filename)).convert('RGB')
        except IOError:
            print(os.path.join(self.root, target_filename))
        target = np.array(target)
        equ = Equirectangular(os.path.join(self.root, target_filename))
        FOV = random.randint(30, 120)
        THETA = random.randint(-180, 180)
        PHI = random.randint(-90, 90)
        img, source, mask = equ.GetPerspective(FOV, THETA, PHI, 512, 1024)
        source = HWC3(source)
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, mask=mask)


class PerspectMask2PanoDataset(Dataset):
    def __init__(self):
        self.data = []
        self.root = 'datasets/Structured3D'
        with open('datasets/Structured3D/all_data_label_train.json', 'rt') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        try:
            target = Image.open(os.path.join(
                self.root, target_filename)).convert('RGB')
        except IOError:
            print(os.path.join(self.root, target_filename))
        target = np.array(target)
        equ = Equirectangular(os.path.join(self.root, source_filename))
        FOV = random.randint(30, 120)
        THETA = random.randint(-180, 180)
        PHI = random.randint(-90, 90)
        img, source, mask = equ.GetPerspective(FOV, THETA, PHI, 512, 1024)
        source = HWC3(source)
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, mask=mask)


class MaskDataset(Dataset):
    def __init__(self, random_rotate=False):
        self.data = []
        self.root = 'datasets/Structured3D'
        with open('datasets/Structured3D/all_data_label_final.json', 'rt') as f:
            self.data = json.load(f)
        self.colroDict = create_color_dict()

    def __len__(self):
        return len(self.data)

    def RGBtoOneHot(self, rgb):
        arr = np.zeros((rgb.shape[0], rgb.shape[1], 1), dtype=np.uint8)
        for label, color in enumerate(self.colroDict.keys()):
            color = np.array(color)
            if label < len(self.colroDict.keys()):
                arr[np.all(rgb == color, axis=-1)] = label
        return arr

    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item['source']
        try:
            source = Image.open(os.path.join(
                self.root, source_filename)).convert('RGB')
        except IOError:
            print(os.path.join(self.root, source_filename))
        source = np.array(source)
        source = self.RGBtoOneHot(source)
        mask_image = Image.fromarray(source.squeeze(), mode='L')
        # datasets/Structured3D/scene_00000/2D_rendering/485145/panorama/full/semantic.png
        save_path = os.path.join(self.root, os.path.split(source_filename)[
                                 0], 'semantic_label.png')
        mask_image.save(save_path)

        return source_filename


class EquirectangularCLIP:
    def __init__(self, img):
        # self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        # self._img = Image.open(img_name).convert('L')
        self._img = np.array(img)
        self._height, self._width = self._img.shape
        self._mask_img = np.zeros((self._height, self._width, 1)) + 41
        self._mask = np.ones((self._height, self._width))

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        # print(self._img.shape)
        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0,  1],
        ], np.float32)
        K_inv = np.linalg.inv(K)

        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate(
            [x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        xyz = xyz @ R.T
        lonlat = xyz2lonlat(xyz)
        XY, XY_INT = lonlat2XY(lonlat, shape=(self._height, self._width, 3))
        XY = XY.astype(np.float32)
        XY_INT = XY_INT.astype(np.int32)
        persp = cv2.remap(
            self._img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        # 遍历XY，将对应位置的Mask置为1
        for i in range(XY.shape[0]):
            for j in range(XY.shape[1]):
                self._mask_img[XY_INT[i][j][1]][XY_INT[i][j][0]
                                                ] = self._img[XY_INT[i][j][1]][XY_INT[i][j][0]]
                self._mask[XY_INT[i][j][1]][XY_INT[i][j][0]] = 0
        return persp, self._mask_img, self._mask


class PerspectMaskCLIP2PanoDataset(Dataset):
    def __init__(self, random_rotate=False, x=0, y=0, z=360):
        self.data = []
        self.root = 'datasets/Structured3D'
        with open('datasets/Structured3D/all_data_label_train.json', 'rt') as f:
            self.data = json.load(f)
        self.random_rotate = random_rotate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item['source']
        source_filename = os.path.join(os.path.split(
            source_filename)[0], 'semantic_label.png')
        target_filename = item['target']
        prompt = item['prompt']
        try:
            target = Image.open(os.path.join(
                self.root, target_filename)).convert('RGB')
        except IOError:
            print(os.path.join(self.root, target_filename))
        target = np.array(target)
        equ_img = Image.open(os.path.join(
            self.root, source_filename)).convert('L')
        equ_img = np.array(equ_img)
        if self.random_rotate:
            X = int(random.random() * self.x)
            Y = int(random.random() * self.y)
            Z = int(random.random() * self.z)
            equ_img = Rot_Equirect(equ_img, (X, Y, Z))
            target = Rot_Equirect(target, (X, Y, Z))
        target = (target.astype(np.float32) / 127.5) - 1.0
        equ = EquirectangularCLIP(equ_img)
        FOV = random.randint(30, 120)
        THETA = random.randint(-180, 180)
        PHI = random.randint(-90, 90)
        img, source, mask = equ.GetPerspective(FOV, THETA, PHI, 512, 1024)
        # Normalize source images to [0, 1].
        # source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].

        return dict(jpg=target, txt=prompt, hint=source, mask=mask)


class NewPerspectMaskCLIP2PanoDataset(Dataset):
    def __init__(self, random_rotate=False, x=0, y=0, z=360):
        self.data = []
        self.root = 'datasets/Structured3D'
        with open('datasets/Structured3D/new_all_data_label_train.json', 'rt') as f:
            self.data = json.load(f)
        self.random_rotate = random_rotate
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item['source']
        source_filename = os.path.join(os.path.split(
            source_filename)[0], 'semantic_label.png')
        target_filename = item['target']
        prompt = item['prompt']
        try:
            target = Image.open(os.path.join(
                self.root, target_filename)).convert('RGB')
        except IOError:
            print(os.path.join(self.root, target_filename))
        target = np.array(target)
        equ_img = Image.open(os.path.join(
            self.root, source_filename)).convert('L')
        equ_img = np.array(equ_img)
        if self.random_rotate:
            X = int(random.random() * self.x)
            Y = int(random.random() * self.y)
            Z = int(random.random() * self.z)
            equ_img = Rot_Equirect(equ_img, (X, Y, Z))
            target = Rot_Equirect(target, (X, Y, Z))
        target = (target.astype(np.float32) / 127.5) - 1.0
        equ = EquirectangularCLIP(equ_img)
        FOV = random.randint(30, 120)
        THETA = random.randint(-180, 180)
        PHI = random.randint(-90, 90)
        img, source, mask = equ.GetPerspective(FOV, THETA, PHI, 512, 1024)
        # Normalize source images to [0, 1].
        # source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].

        return dict(jpg=target, txt=prompt, hint=source, mask=mask)

class NewMaskCLIP2PanoDataset(Dataset):
    def __init__(self, random_rotate=False, x=0, y=0, z=360):
        self.data = []
        self.root = 'datasets/Structured3D'
        with open('datasets/Structured3D/new_all_data_label_train.json', 'rt') as f:
            self.data = json.load(f)
        self.random_rotate = random_rotate
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item['source']
        source_filename = os.path.join(os.path.split(
            source_filename)[0], 'semantic_label.png')
        target_filename = item['target']
        prompt = item['prompt']
        try:
            target = Image.open(os.path.join(
                self.root, target_filename)).convert('RGB')
        except IOError:
            print(os.path.join(self.root, target_filename))
        target = np.array(target)
        equ_img = Image.open(os.path.join(
            self.root, source_filename)).convert('L')
        equ_img = np.array(equ_img)
        if self.random_rotate:
            X = int(random.random() * self.x)
            Y = int(random.random() * self.y)
            Z = int(random.random() * self.z)
            equ_img = Rot_Equirect(equ_img, (X, Y, Z))
            target = Rot_Equirect(target, (X, Y, Z))
        target = (target.astype(np.float32) / 127.5) - 1.0
        source = np.array(equ_img)
        mask = np.ones((source.shape[0], source.shape[1]))
        # Normalize source images to [0, 1].
        # source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].

        return dict(jpg=target, txt=prompt, hint=source, mask=mask)
