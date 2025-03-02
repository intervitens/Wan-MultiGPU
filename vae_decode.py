import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pickle
from datetime import datetime
import argparse

import torch
from wan.utils.utils import cache_video
from wan.modules.vae import WanVAE

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--latents_file",
        type=str,
        default="x0.pkl",
        help="The path to the saved latents.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")

    args = parser.parse_args()
    return args

def vae_decode(args):
    vae_stride = (4, 8, 8)
    patch_size = (1, 2, 2)

    vae_checkpoint = 'Wan2.1_VAE.pth'
    vae = WanVAE(
        vae_pth=os.path.join(args.ckpt_dir, vae_checkpoint),
        device="cuda:0")

    with open(args.latents_file, 'rb') as file:
        x0 = [pickle.load(file).to("cuda:0")]

    video = vae.decode(x0)[0]

    if args.save_file is None:
        args.save_file = "out_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"

    cache_video(
        tensor=video[None],
        save_file=args.save_file,
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1))


if __name__ == "__main__":
    args = _parse_args()
    vae_decode(args)






