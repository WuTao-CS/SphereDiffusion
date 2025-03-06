from PIL import Image
import numpy as np
import os
import json
from tqdm import tqdm
from equirect_rotation_fast import Rot_Equirect
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ControlNet')
    # checkpoint and log
    parser.add_argument('--sample_dir', type=str, help='input image data dir', default=None)
    parser.add_argument('--sample_file', type=str, help='input image data json', default=None)
    parser.add_argument('--save_path', type=str, help='output npz file path')

    args = parser.parse_args()
    return args


def create_rotate_npz_from_json(json_file_name="datasets/Structured3D/all_data_label_test.json", save_path="",x=3,y=3,z=180):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    root = 'datasets/Structured3D'
    with open(json_file_name, 'rt') as f:
        all_data = json.load(f)
    samples = []
    for item in tqdm(all_data, desc="Building .npz file from samples"):
        sample_pil = Image.open(os.path.join(
            root, item['target'])).convert('RGB')
        sample_np = np.asarray(sample_pil)
        sample_np = Rot_Equirect(sample_np, (x, y, z)).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    print(samples.shape)
    np.savez(save_path, arr_0=samples)
    print(f"Saved npz file to ", save_path)
    return


def create_npz_from_json(json_file_name="datasets/Structured3D/all_data_label_test.json", save_path=""):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    root = 'datasets/Structured3D'
    with open(json_file_name, 'rt') as f:
        all_data = json.load(f)
    samples = []
    for item in tqdm(all_data, desc="Building .npz file from samples"):
        sample_pil = Image.open(os.path.join(
            root, item['target'])).convert('RGB')
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    print(samples.shape)
    np.savez(save_path, arr_0=samples)
    print(f"Saved npz file to ", save_path)
    return


def create_npz_from_sample_folder(sample_dir, save_path=""):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    filenames = os.listdir(sample_dir)
    for name in tqdm(filenames, desc="Building .npz file from samples"):
        if not name.endswith('.png'):
            continue
        sample_pil = Image.open(
            os.path.join(sample_dir, name)).convert('RGB')
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    npz_path = f"{sample_dir}.npz" if save_path == "" else save_path

    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return


def create_connect_npz_from_sample_folder(sample_dir, save_path=""):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    filenames = os.listdir(sample_dir)
    for name in tqdm(filenames, desc="Building .npz file from samples"):
        if not name.endswith('.png'):
            continue
        sample_pil = Image.open(
            os.path.join(sample_dir, name)).convert('RGB')
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(np.concatenate(
            (sample_np[:, -128:, :], sample_np[:, 128:, :]), axis=1))
    samples = np.stack(samples)
    npz_path = f"{sample_dir}.npz" if save_path == "" else save_path

    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return


args = parse_args()
if args.sample_file is not None:
    create_npz_from_json(json_file_name=args.sample_file,save_path=args.save_path)
elif args.sample_dir is not None:
    create_npz_from_sample_folder(sample_dir=args.sample_dir, save_path=args.save_path)

