import torch
import math
import os
import sys
import time
from PIL import Image
import numpy as np

def Rot_Matrix(rotation, unit='degree', device='cpu'):
    rotation = torch.tensor(rotation, dtype=torch.float32, device=device)

    if(unit == 'degree'):
        rotation = torch.deg2rad(rotation)
    elif(unit != 'rad'):
        print("ParameterError: "+unit+"is wrong unit!")
        return
    Rx = torch.tensor([[1, 0, 0], [0, math.cos(rotation[0]), -math.sin(rotation[0])],
                   [0, math.sin(rotation[0]), math.cos(rotation[0])]], device=device)
    Ry = torch.tensor([[math.cos(rotation[1]), 0, math.sin(rotation[1])],
                   [0, 1, 0], [-math.sin(rotation[1]), 0, math.cos(rotation[1])]], device=device)
    Rz = torch.tensor([[math.cos(rotation[2]), -math.sin(rotation[2]), 0],
                   [math.sin(rotation[2]), math.cos(rotation[2]), 0], [0, 0, 1]], device=device)
    R = torch.matmul(torch.matmul(Rx, Ry), Rz)
    return R


def Pixel2LonLat(equirect, device='cpu'):
    W = equirect.shape[1]
    H = equirect.shape[0]
    Lon = torch.tensor([2*(x/W-0.5)*math.pi for x in range(W)], device=device)
    Lat = torch.tensor([(0.5-y/H)*math.pi for y in range(H)], device=device)

    Lon = Lon.repeat(H, 1)
    Lat = Lat.view(H, 1).repeat(1, W)

    LonLat = torch.stack((Lon, Lat), dim=-1)
    return LonLat


def LonLat2Sphere(LonLat):
    x = torch.cos(LonLat[:, :, 1])*torch.cos(LonLat[:, :, 0])
    y = torch.cos(LonLat[:, :, 1])*torch.sin(LonLat[:, :, 0])
    z = torch.sin(LonLat[:, :, 1])

    xyz = torch.stack((x, y, z), dim=-1)
    return xyz


def Sphere2LonLat(xyz):
    Lon = torch.atan2(xyz[:, :, 1], xyz[:, :, 0])
    Lat = math.pi/2 - torch.acos(xyz[:, :, 2])

    LonLat = torch.stack((Lon, Lat), dim=-1)
    return LonLat


def LonLat2Pixel(LonLat):
    width = LonLat.shape[1]
    height = LonLat.shape[0]
    j = (width*(LonLat[:, :, 0]/(2*math.pi)+0.5)) % width
    i = (height*(0.5-(LonLat[:, :, 1]/math.pi))) % height

    ij = torch.stack((i, j), dim=-1).long()
    return ij

def proccesing(src, src_Pixel):
    out = torch.zeros_like(src)
    for i in range(src.shape[0]):
        out[i] = src[i][src_Pixel[:, :, 0], src_Pixel[:, :, 1]]
    return out


def Rot_Equirect(src, rotation=(0, 0, 0), device='cpu'):
    R = Rot_Matrix(rotation, device=device)

    out_LonLat = Pixel2LonLat(src[0], device=device)
    out_xyz = LonLat2Sphere(out_LonLat)

    src_xyz = torch.zeros_like(out_xyz)
    src_xyz = torch.einsum("ka,ijk->ija", R, out_xyz)

    src_LonLat = Sphere2LonLat(src_xyz)
    src_Pixel = LonLat2Pixel(src_LonLat)

    out = proccesing(src, src_Pixel)
    return out

if __name__ == "__main__":
    src = Image.open('source.png')
    src = np.array(src)
    start = time.time()
    src = torch.tensor(src)
    src = src.repeat(5, 1, 1, 1)
    out = Rot_Equirect(src, (175,175,90),device='cuda').cpu().numpy()
    print(time.time()-start)
    Image.fromarray(out[-1]).save('out.png')

# if __name__ == "__main__":
#     src = Image.open('out.png')
#     src = np.array(src)
#     start = time.time()
#     out = Rot_Equirect(src, (-5,-5, -90),device='cuda')
#     print(time.time()-start)
#     Image.fromarray(out).save('back.png')

# python annotator/equrotation.py datasets/Structured3D/scene_00000/2D_rendering/485142/panorama/full/rgb_rawlight.png 5 5 180 out.png
# python annotator/equrotation.py out.png -5 -5 -180 back.png