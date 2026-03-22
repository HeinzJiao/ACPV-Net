#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encode images, masks, or heatmaps into first-stage autoencoder latents.

This script loads a pretrained first-stage autoencoder, estimates or uses a
given latent scale factor, and saves encoded latent tensors in `.pt` format.
Optional reconstruction and latent-channel visualization are also supported.

Usage example:
  python latent_encoder_accelerated.py \
    --config /path/to/autoencoder_config.yaml \
    --input /path/to/input_dir \
    --output_dir /path/to/output_dir \
    --type heatmap
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution


def disabled_train(self, mode=True):
    """Keep module in eval() even if someone calls train()."""
    return self


class AutoencoderLatentEncoder(torch.nn.Module):
    """
    Thin wrapper around AutoencoderKL.

    It provides:
      - encoding from input tensors in [-1, 1] to scaled latent tensors
      - optional decoding for sanity checks
    """
    def __init__(self, ae_config, scale_factor: float):
        super().__init__()
        self.scale_factor = float(scale_factor)
        self.ae = instantiate_from_config(ae_config).eval()
        self.ae.train = disabled_train
        for p in self.ae.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.ae.encode(x)
        if isinstance(enc, DiagonalGaussianDistribution):
            z = enc.sample()
        elif isinstance(enc, torch.Tensor):
            z = enc
        else:
            raise TypeError(f"Unsupported encoder output: {type(enc)}")
        return (self.scale_factor * z).detach()

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = (1.0 / self.scale_factor) * z
        xrec = self.ae.decode(z)
        return xrec


def _is_valid_file(path: str, input_type: str) -> bool:
    p = path.lower()
    if input_type in ["mask", "image"]:
        return p.endswith(".png") or p.endswith(".jpg") or p.endswith(".jpeg")
    if input_type == "heatmap":
        return p.endswith(".npy")
    return False


def _load_as_chw_tensor(path: str, input_type: str) -> torch.Tensor:
    """
    Return CHW float32 tensor in [-1, 1].

    - image: read as RGB
    - mask: read as grayscale, normalize, then repeat to 3 channels
    - heatmap: read as float `.npy`, map to [-1, 1], then repeat to 3 channels
    """
    if input_type == "image":
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # CHW
    elif input_type == "mask":
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        arr = arr * 2.0 - 1.0
        return torch.from_numpy(arr)[None, ...].repeat(3, 1, 1)  # 3,H,W
    else:
        heat = np.load(path).astype(np.float32)  # H,W
        heat = heat * 2.0 - 1.0
        ten = torch.from_numpy(heat)[None, ...].repeat(3, 1, 1)  # 3,H,W
        return ten


class SimpleInputDataset(Dataset):
    def __init__(self, files, input_type):
        self.files = files
        self.input_type = input_type

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        x = _load_as_chw_tensor(p, self.input_type)
        return x, p


@torch.no_grad()
def _estimate_scale_factor(ae_config, loader, device, max_images: int) -> float:
    """
    Estimate `scale_factor = 1 / std(z)` using the autoencoder with
    `scale_factor = 1.0`.

    If `max_images > 0`, only the first `max_images` samples are used.
    """
    ae = instantiate_from_config(ae_config).to(device).eval()
    ae.train = disabled_train
    for p in ae.parameters():
        p.requires_grad = False

    total = 0
    s1 = 0.0
    s2 = 0.0
    seen = 0

    for batch, _ in loader:
        batch = batch.to(device, non_blocking=True)
        enc = ae.encode(batch)
        if isinstance(enc, DiagonalGaussianDistribution):
            z = enc.sample()
        else:
            z = enc
        z = z.float()
        s1 += z.sum().item()
        s2 += (z * z).sum().item()
        total += z.numel()
        seen += batch.size(0)
        if max_images > 0 and seen >= max_images:
            break

    mean = s1 / max(total, 1)
    var = (s2 / max(total, 1)) - mean * mean
    var = max(var, 1e-12)
    std = var ** 0.5
    return 1.0 / std


def _save_latent_vis(z: torch.Tensor, out_dir: str, vis_channels: int):
    """
    Save the first K latent channels as JET-colored PNGs.

    Args:
        z: Latent tensor of shape [C, H, W] on CPU
        out_dir: Output directory
        vis_channels: Number of channels to visualize
    """
    os.makedirs(out_dir, exist_ok=True)
    z = z.float().clamp(-1.0, 1.0)
    K = min(vis_channels, z.size(0))
    z_u8 = ((z[:K] + 1.0) / 2.0 * 255.0).to(torch.uint8).numpy()  # K,H,W

    for c in range(K):
        gray = z_u8[c]
        bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(os.path.join(out_dir, f"latent_channel_{c}.png"))


def main():
    parser = argparse.ArgumentParser("Encode image/mask/heatmap into AutoencoderKL latents (batched).")
    parser.add_argument("--config", type=str, required=True, help="autoencoder config yaml (decoder-style)")
    parser.add_argument("--input", type=str, required=True, help="input file or directory")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory")

    parser.add_argument("--type", type=str, required=True, choices=["mask", "image", "heatmap"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--scale_factor", type=float, default=-1.0, help="if >0, use this scale factor; otherwise estimate")
    parser.add_argument("--scale-samples", type=int, default=128, help="samples used to estimate scale_factor (0=all)")

    parser.add_argument("--reconstruct", action="store_true", help="decode latents back to image-space for sanity check")
    parser.add_argument("--save-z-vis", action="store_true", help="save latent channel visualizations (JET)")
    parser.add_argument("--vis-channels", type=int, default=4)

    args = parser.parse_args()

    # Match latent_decoder_accelerated.py by reading cfg.model.params.first_stage_config.
    with open(args.config, "r") as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    ae_config = cfg.model.params.first_stage_config

    if os.path.isdir(args.input):
        files = sorted([
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if _is_valid_file(os.path.join(args.input, f), args.type)
        ])
    else:
        files = [args.input]

    print(f"Found {len(files)} files.")
    os.makedirs(args.output_dir, exist_ok=True)

    ds = SimpleInputDataset(files, args.type)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=True, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    if args.scale_factor and args.scale_factor > 0:
        scale_factor = float(args.scale_factor)
    else:
        scale_factor = _estimate_scale_factor(
            ae_config=ae_config,
            loader=loader,
            device=device,
            max_images=int(args.scale_samples)
        )

    with open(os.path.join(args.output_dir, "scale_factor.txt"), "w") as f:
        f.write(f"{scale_factor:.6f}\n")
    print(f"Using scale_factor = {scale_factor:.6f}")

    model = AutoencoderLatentEncoder(ae_config, scale_factor=scale_factor).to(device).eval()

    out_latent = os.path.join(args.output_dir, "z")
    out_vis = os.path.join(args.output_dir, "z_vis")
    out_rec = os.path.join(args.output_dir, "xrec")
    os.makedirs(out_latent, exist_ok=True)

    pbar = tqdm(total=len(files), desc="Encoding", unit="img")
    for batch, paths in loader:
        batch = batch.to(device, non_blocking=True).float()

        z = model.encode(batch)  # B,C,h,w
        xrec = model.decode(z) if args.reconstruct else None

        z_cpu = z.detach().cpu()
        for zi, p in zip(z_cpu, paths):
            name = os.path.splitext(os.path.basename(p))[0]
            torch.save(zi, os.path.join(out_latent, f"{name}.pt"))

            if args.save_z_vis:
                _save_latent_vis(zi, os.path.join(out_vis, name), args.vis_channels)

        if args.reconstruct and xrec is not None:
            os.makedirs(out_rec, exist_ok=True)

            # Save reconstruction visualizations.
            x = xrec.detach().cpu().clamp(-1.0, 1.0).cpu().numpy()              # [B,3,H,W]
            x = ((x + 1.0) / 2.0 * 255.0).astype(np.uint8)
            x = np.transpose(x, (0, 2, 3, 1))                  # [B,H,W,3] RGB

            for img_rgb, p in zip(x, paths):
                name = os.path.splitext(os.path.basename(p))[0]
                if args.type == "heatmap":
                    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)  # BGR
                    cv2.imwrite(os.path.join(out_rec, f"{name}.png"), heat)
                else:
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(out_rec, f"{name}.png"), img_bgr)

        pbar.update(len(paths))
    pbar.close()
    print("All done.")


if __name__ == "__main__":
    main()

    """
    Usage (Example):
    PYTHONPATH=./:$PYTHONPATH python latent_encoder_accelerated.py \
    --config config-files/autoencoder_kl_f4.yaml \
    --input ../../agricultural_parcel_extraction/AI4SmallFarms_split_0.05/train_vertex_heatmaps_sigma-3 \
    --output_dir ../../agricultural_parcel_extraction/AI4SmallFarms_split_0.05/train_vertex_heatmaps_sigma-3_latent \
    --type heatmap \
    --batch-size 4 \
    --num-workers 8 \
    --scale-samples 128 \
    --reconstruct \
    --save-z-vis
    """
