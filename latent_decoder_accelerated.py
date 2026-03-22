#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decode first-stage autoencoder latents back to pixel space.

This script loads latent tensors saved in `.pt` format, decodes them with a
pretrained first-stage autoencoder, and optionally saves visualization images
as RGB outputs or JET-colored heatmaps.

Usage example:
  python latent_decoder_accelerated.py \
    --config /path/to/autoencoder_config.yaml \
    --latent_dir /path/to/latent_dir \
    --output_dir /path/to/output_dir \
    --npy_dir /path/to/npy_dir \
    --scale_factor 0.18215
"""

import os
import argparse
import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


def disabled_train(self, mode=True):
    """Disable train/eval mode toggling for frozen models."""
    return self


def is_valid_pt(path: str) -> bool:
    """Return True if `path` is an existing .pt file."""
    return path.lower().endswith(".pt") and os.path.isfile(path)


class LatentDecoder(nn.Module):
    """
    Lightweight wrapper around the first-stage autoencoder.

    Important:
      - `z` is divided by `scale_factor` before decoding, matching the
        encoding pipeline.
      - The instantiated autoencoder is frozen and used in eval mode.
    """

    def __init__(self, config, scale_factor: float):
        super().__init__()
        self.scale_factor = float(scale_factor)

        # Instantiate the first-stage autoencoder from config.
        self.model = instantiate_from_config(config)
        self.model.eval()
        self.model.train = disabled_train

        # Freeze all parameters.
        for p in self.model.parameters():
            p.requires_grad = False

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents into pixel-space reconstruction.

        Args:
            z: [B, C, H, W] latent tensor

        Returns:
            xrec: [B, 3, H', W'] reconstructed tensor (in [-1, 1])
        """
        z = z / self.scale_factor
        return self.model.decode(z)


# -----------------------------
# Dataset / DataLoader
# -----------------------------
class LatentPTDataset(Dataset):
    """
    Loads latent tensors from .pt files.

    Each file is expected to store a tensor shaped:
      - [C, H, W] (preferred), or
      - [1, C, H, W] (will be squeezed to [C, H, W])
    """

    def __init__(self, pt_files):
        self.files = pt_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        z = torch.load(p)
        if z.ndim == 3:
            # [C, H, W]
            pass
        elif z.ndim == 4 and z.shape[0] == 1:
            # [1, C, H, W] -> [C, H, W]
            z = z.squeeze(0)
        else:
            raise ValueError(f"Unexpected latent shape in {p}: {tuple(z.shape)}")

        return z.float(), p  # Return the CHW tensor and source path.


@torch.no_grad()
def process_batch(latents: torch.Tensor,
                  paths,
                  decoder: LatentDecoder,
                  output_img_dir: str,
                  output_npy_dir: str,
                  vis_type: str):
    """
    Decode a batch of latents and save results.

    Args:
        latents: [B, C, H, W]
        paths:   List[str], aligned with the batch order
        decoder: LatentDecoder instance
        output_img_dir: Directory for visualization PNGs
        output_npy_dir: Directory for decoded tensors saved as `.npy`
        vis_type: "mask" for direct RGB visualization, "heatmap" for JET colormap

    Outputs:
      - `.npy`: decoded float tensor in CHW format
      - `.png`: visualization image
    """
    xrec = decoder.decode(latents)  # [B,3,H,W], in [-1, 1]

    os.makedirs(output_npy_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)

    # Save reconstructed tensors as `.npy` in CHW format.
    for xr, p in zip(xrec, paths):
        name = os.path.splitext(os.path.basename(p))[0]
        np.save(os.path.join(output_npy_dir, f"{name}.npy"), xr.cpu().numpy())

    # Save visualization PNGs.
    x = xrec.clamp(-1., 1.).cpu().numpy()              # [B,3,H,W]
    x = ((x + 1.0) / 2.0 * 255.0).astype(np.uint8)
    x = np.transpose(x, (0, 2, 3, 1))                  # [B,H,W,3] RGB

    for img_rgb, p in zip(x, paths):
        name = os.path.splitext(os.path.basename(p))[0]
        if vis_type == "heatmap":
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)  # BGR
            cv2.imwrite(os.path.join(output_img_dir, f"{name}.png"), heat)
        else:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_img_dir, f"{name}.png"), img_bgr)


def main():
    parser = argparse.ArgumentParser(description="Batched latent decoding & visualization")
    parser.add_argument("--config", type=str, required=True, help="YAML of first_stage autoencoder")
    parser.add_argument("--latent_dir", type=str, required=True, help="Directory of .pt latents")
    parser.add_argument("--output_dir", type=str, required=True, help="Dir to save decoded images")
    parser.add_argument("--npy_dir", type=str, required=True, help="Dir to save reconstructed .npy")
    parser.add_argument("--scale_factor", type=float, required=True, help="Scale factor used in encoding")
    parser.add_argument("--type", type=str, default="heatmap", choices=["mask", "heatmap"],
                        help="Visualization type: mask -> direct RGB, heatmap -> JET")

    # Performance-related settings.
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # Load the first-stage autoencoder config.
    with open(args.config, "r") as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    first_stage_cfg = cfg.model.params.first_stage_config

    # Collect latent `.pt` files.
    pt_files = sorted([os.path.join(args.latent_dir, f)
                       for f in os.listdir(args.latent_dir)
                       if is_valid_pt(os.path.join(args.latent_dir, f))])
    print(f"Found {len(pt_files)} latent .pt files in: {args.latent_dir}")

    if len(pt_files) == 0:
        print("No .pt files found. Exit.")
        return

    # Build the DataLoader.
    ds = LatentPTDataset(pt_files)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch], dim=0),  # [B,C,H,W]
            [b[1] for b in batch]                       # list of file paths
        )
    )

    # Instantiate the decoder.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    decoder = LatentDecoder(first_stage_cfg, args.scale_factor).to(device).eval()

    # Batched decoding.
    pbar = tqdm(total=len(pt_files), unit="file", desc="Decoding")
    for latents, paths in loader:
        latents = latents.to(device, non_blocking=True)
        process_batch(
            latents=latents,
            paths=paths,
            decoder=decoder,
            output_img_dir=args.output_dir,
            output_npy_dir=args.npy_dir,
            vis_type=args.type,
        )
        pbar.update(len(paths))
    pbar.close()

    print("All done.")


if __name__ == "__main__":
    main()
