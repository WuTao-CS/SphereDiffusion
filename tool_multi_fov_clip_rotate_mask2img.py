from share import *
import config
import einops
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import argparse
from PIL import Image
import os
import json
from tutorial_dataset import EquirectangularCLIP
from annotator.equirect_rotation_fast import Rot_Equirect

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, rotation_step=4):
    with torch.no_grad():
        H, W, _ = input_image.shape

        control = torch.from_numpy(input_image.copy()).cuda()
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [
            model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [
            model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
            [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.rotation_sample(ddim_steps, num_samples,
                                                              shape, cond, verbose=True, eta=eta,
                                                              unconditional_guidance_scale=scale,
                                                              unconditional_conditioning=un_cond, rotation_step=rotation_step)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c')
                     * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [input_image] + results


def read_json_list(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        try:
            school_list = json.load(f)
        except:
            school_list = []
    return school_list


def parse_args():
    parser = argparse.ArgumentParser(description='ControlNet')
    # checkpoint and log
    parser.add_argument('--data', type=str, default='all',
                        help='test_data_mode all or tiny')
    parser.add_argument('--resolution', type=int, default=512,
                        help='shot image resolution end')
    parser.add_argument('--strength', type=float, default=1.0,
                        help='shot image end')
    parser.add_argument('--step', type=int, default=64,
                        help='ddim_steps')
    parser.add_argument('--seed', type=int, default=-1,
                        help='seed')
    parser.add_argument('--fov', type=int, default=90,
                        help='fov')
    parser.add_argument('--rotate-step', type=int, default=4,
                        help='seed')
    parser.add_argument('--config-file', metavar="FILE",
                        default='./models/cldm_v15_clip_new_resize.yaml',
                        help='config file path')
    parser.add_argument('--resume', type=str, default='./perspect_resize_clip_mask_checkpoints_zero_conv/ControlNet-epoch=29-global_step=103289.0.ckpt',
                        help='path to model path')
    parser.add_argument('--save-path', type=str, default='./fov_clip_zero_conv/',
                        help='path to save inference result')
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--rotate', action='store_true', default=False)
    parser.add_argument('--x', type=int, default=1)
    parser.add_argument('--y', type=int, default=1)
    parser.add_argument('--z', type=int, default=180)
    args = parser.parse_args()
    return args


args = parse_args()
model = create_model(args.config_file).cpu()
model.load_state_dict(load_state_dict(args.resume, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)
num_samples = 1
image_resolution = args.resolution
strength = args.strength
guess_mode = False
ddim_steps = args.step
scale = 9.0
seed = args.seed
eta = 0.0
a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
if args.data == 'all':
    print("Using all data")
    test_list = read_json_list(
        'datasets/Structured3D/all_data_label_test.json')
else:
    print("Using tiny data")
    test_list = read_json_list(
    'datasets/Structured3D/tiny_data_test.json')
root = 'datasets/Structured3D'
total_num = 8
split = len(test_list)/total_num
print('total num {}, split {}'.format(total_num, split))
start = int((args.sample-1)*split)
end = int(args.sample*split)
cnt = start
rotation_step = args.rotate_step
print("rotation step {}".format(rotation_step))
if os.path.isdir(args.save_path) is False:
    os.mkdir(args.save_path)
if args.rotate:
    print('rotate angle {},{},{}'.format(args.x, args.y, args.z))
print(args.fov)
for item in test_list[start:end]:
    source_filename = os.path.join(os.path.split(
        item["source"])[0], 'semantic_label.png')
    equ_img = Image.open(os.path.join(
        root, source_filename)).convert('L')
    equ_img = np.array(equ_img)
    equ = EquirectangularCLIP(equ_img)
    FOV = args.fov
    THETA = 0
    PHI = 0
    img, source, mask = equ.GetPerspective(FOV, THETA, PHI, 512, 1024)
    if args.rotate:
        source = Rot_Equirect(source, (args.x, args.y, args.z))
    input_image = source
    prompt = item["prompt"]
    ips = process(input_image, prompt, a_prompt, n_prompt, num_samples,
                  image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, rotation_step)
    res = Image.fromarray(ips[1]).save(
        os.path.join(args.save_path, 'res_{}_{}.png'.format(args.sample,cnt)))
    cnt += 1
