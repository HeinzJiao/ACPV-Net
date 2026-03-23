#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize vertex heatmaps stored as `.npy` arrays.

This script converts heatmaps to grayscale or JET-colored PNG images for quick
inspection.

Example:
    python tools/visualize_vertex_heatmaps.py \
        --input_dir /path/to/heatmaps \
        --output_dir /path/to/heatmap_vis
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse

def visualize_heatmaps(input_dir, output_dir, mode='gray'):
    """
    Convert all vertex heatmaps in a folder to visualization images.

    Args:
        input_dir (str): Folder containing `.npy` heatmaps of shape `(H, W)`.
        output_dir (str): Folder to save visualization images.
        mode (str): Visualization mode, either `"gray"` or `"jet"`.
    """
    os.makedirs(output_dir, exist_ok=True)

    heatmap_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    for fname in tqdm(heatmap_files, desc="Visualizing heatmaps"):
        path = os.path.join(input_dir, fname)
        heatmap = np.load(path)

        # Map the heatmap to image intensity range.
        heatmap_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)

        # Render the selected visualization mode.
        if mode == 'gray':
            vis_img = heatmap_uint8
        elif mode == 'jet':
            vis_img = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        else:
            raise ValueError(f"Unsupported mode: {mode}, choose 'gray' or 'jet'.")

        # Save the visualization.
        out_path = os.path.join(output_dir, fname.replace('.npy', '.png'))
        cv2.imwrite(out_path, vis_img)

    print(f"Processed {len(heatmap_files)} heatmap files. Results saved to:\n{output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize junction heatmaps.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing .npy heatmap files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save visualized heatmap images.')
    parser.add_argument('--resize', type=int, nargs=2, default=None, help='Resize heatmap to given size (H, W).')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    visualize_heatmaps(args.input_dir, args.output_dir, mode='jet')
