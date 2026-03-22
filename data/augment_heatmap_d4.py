#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply D4 augmentations to 2D heatmaps or mask-like arrays.

This script reads `.npy` or `.png` files from an input directory and writes
the transformed results into per-transform subdirectories.

Usage example:
  python data/augment_heatmap_d4.py \
    --input_dir /path/to/input_dir \
    --output_dir /path/to/output_dir \
    --transforms flip_v flip_h
"""

import os
import numpy as np
import argparse
from PIL import Image

ALL_D4_TRANSFORMS = (
    "rot0",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "flip_diag",
    "flip_anti_diag",
)


def apply_d4_transforms(arr, selected_transforms):
    """
    Apply the eight unique D4 transforms to a 2D array.

    Returns:
        dict: Mapping from transform name to transformed array.
    """
    all_transforms = {
        "rot0": arr,
        "rot90": np.rot90(arr, k=1),
        "rot180": np.rot90(arr, k=2),
        "rot270": np.rot90(arr, k=3),
        "flip_h": np.flip(arr, axis=1),               # Horizontal flip
        "flip_v": np.flip(arr, axis=0),               # Vertical flip
        "flip_diag": np.transpose(arr),               # Main-diagonal flip
        "flip_anti_diag": np.transpose(np.flip(arr, axis=(0,1))) # Anti-diagonal flip
    }
    return {name: all_transforms[name] for name in selected_transforms}


def load_array(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path), ext
    if ext == ".png":
        return np.array(Image.open(path)), ext
    raise ValueError(f"Unsupported file type: {path}")


def save_array(path, arr, ext):
    if ext == ".npy":
        np.save(path, arr)
    elif ext == ".png":
        Image.fromarray(arr).save(path)
    else:
        raise ValueError(f"Unsupported output type: {ext}")


def process_folder(input_dir, output_dir, selected_transforms):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith(".npy") or f.endswith(".png")
    )

    for fname in files:
        heatmap, ext = load_array(os.path.join(input_dir, fname))
        transforms = apply_d4_transforms(heatmap, selected_transforms)
        base_name = os.path.splitext(fname)[0]

        for tname, transformed in transforms.items():
            out_dir_t = os.path.join(output_dir, tname)
            os.makedirs(out_dir_t, exist_ok=True)
            out_path = os.path.join(out_dir_t, f"{base_name}{ext}")
            save_array(out_path, transformed, ext)

        print(f"Processed: {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Input directory containing .npy or .png files.")
    parser.add_argument("--output_dir", required=True, help="Output directory for transformed results.")
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=list(ALL_D4_TRANSFORMS),
        choices=ALL_D4_TRANSFORMS,
        help="D4 transforms to apply. Defaults to all eight transforms.",
    )
    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir, args.transforms)
