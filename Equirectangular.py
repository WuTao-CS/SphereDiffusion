import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import random


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


def create_color_palette():
    return [
        (0, 0, 0),              # unknown
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


class EquirectangularCLIP:
    def __init__(self, img_name):
        # self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        self._img = Image.open(img_name).convert('RGB')
        self._img = np.array(self._img)
        self._height, self._width, _ = self._img.shape
        self._mask_img = np.zeros((self._height, self._width, 3), np.uint8)
        self._mask = np.ones((self._height, self._width))
        self.emb = np.load("class_emb.npy")
        vector = np.reshape(emb[-1], (1, 1, 768))
        self._mask_clip_img = np.broadcast_to(
            vector, (self._height, self._width, 768))
        self._mask_clip_img.flags.writeable = True
        self.color = create_color_palette()

    def GetTextEmd(self, mask):
        for idx in range(len(self.color)):
            if np.array_equal(mask, np.array(self.color[idx])):
                return emb[idx]

    def GetPerspective(self, FOV, THETA, PHI, height, width):

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
                self._mask[XY_INT[i][j][1]][XY_INT[i][j][0]] = 0
                self._mask_clip_img[XY_INT[i][j][1]][XY_INT[i][j]
                                                     [0]] = GetTextEmd(self._img[XY_INT[i][j][1]][XY_INT[i][j][0]])
        return persp, self._mask_clip_img, self._mask
