import argparse

import torch
from torchvision import utils
from tqdm import tqdm
import os
import json

from model import Generator


def generate(args, g_ema, device):
    sample_path = f"{args.path}/generated"
    last_name = [p for p in os.listdir(sample_path) if p.endswith("png")][-1]
    start = int(os.path.basename(last_name)) + 1
    with torch.no_grad(), open("idx.jsonl", "a") as fp:
        g_ema.eval()
        for i in tqdm(range(start, start + args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, _ = g_ema([sample_z])

            image_id = str(i).zfill(6)

            utils.save_image(
                sample,
                f"{sample_path}/{image_id}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

            print(json.dumps({"id": image_id, "sample_z": sample_z}), file=fp)


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()

    parser.add_argument("path", type=str)
    parser.add_argument("ckpt", type=str)

    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--pics", type=int, default=20)

    parser.add_argument("--channel_multiplier", type=int, default=2)

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(f"{args.path}/checkpoints/{args.ckpt}")

    g_ema.load_state_dict(checkpoint["g_ema"])

    generate(args, g_ema, device)
