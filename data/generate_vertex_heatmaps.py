#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate vertex heatmaps from per-image vertex JSON files.

Each input JSON file is expected to contain a list of vertex coordinates.
The script converts those vertices into a Gaussian heatmap and saves the
result as a `.npy` file.

Usage example:
  python data/generate_vertex_heatmaps.py \
    --json_dir /path/to/json_dir \
    --save_dir /path/to/save_dir \
    --sigma 3
"""

import os
import json
import numpy as np
import argparse
from tqdm import tqdm
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="Generate junction heatmaps from vertex JSON files.")
    parser.add_argument('--json_dir', type=str, required=True, help='Directory containing junction JSON files.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save heatmap .npy files.')
    parser.add_argument('--sigma', type=float, default=2.5, help='Gaussian sigma for heatmap generation.')
    return parser.parse_args()

# Core utilities for 2D Gaussian heatmap generation.

def gaussian_2d(x, y, x0, y0, sigma):
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

def generate_heatmap(vertex_locations, heatmap_shape, sigma=2.5):
    heatmap = np.zeros(heatmap_shape, dtype=np.float32)

    if len(vertex_locations) == 0:
        return heatmap

    # Build pixel grid: x for width (cols), y for height (rows)
    x = np.arange(heatmap_shape[1])
    y = np.arange(heatmap_shape[0])
    x, y = np.meshgrid(x, y)

    for loc in vertex_locations:
        heatmap = np.maximum(heatmap, gaussian_2d(x, y, loc[0], loc[1], sigma))

    max_value = np.max(heatmap)
    if max_value > 0:
        heatmap = heatmap / max_value

    heatmap[heatmap < 1e-8] = 0
    return heatmap

# Main entry point.

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    json_files = [f for f in os.listdir(args.json_dir) if f.endswith(".json")]
    for json_file in tqdm(json_files, desc="Generating heatmaps"):
        name = os.path.splitext(json_file)[0]
        json_path = os.path.join(args.json_dir, json_file)

        # Load vertex coordinates.
        with open(json_path, 'r') as f:
            vertices = json.load(f)

        image_width, image_height = 512, 512

        # Clip coordinates to valid image bounds.
        clipped_vertices = []
        for v in vertices:
            x, y = v[:2]
            x_clipped = np.clip(x, 0, image_width - 1)
            y_clipped = np.clip(y, 0, image_height - 1)
            clipped_vertices.append([x_clipped, y_clipped])

        # Generate the heatmap.
        heatmap = generate_heatmap(clipped_vertices, (image_height, image_width), sigma=args.sigma)

        # Save as .npy.
        save_path = os.path.join(args.save_dir, f"{name}.npy")
        np.save(save_path, heatmap)

if __name__ == "__main__":
    main()
    """
    python generate_junction_heatmaps.py \
        --json_dir ./deventer_512/train/junctions \
        --save_dir ./deventer_512/train/junction_heatmaps_sigma-3 \
        --sigma 3
        
    python ./data/generate_junction_heatmaps.py \
        --json_dir ./data/deventer_512/poly_gt_global_boundary/train/d-2_angle_tol_deg-10-5-2_corner_eps-2_min_sep-3_len_px-10-6_v5/final_vertices \
        --save_dir ./data/deventer_512/poly_gt_global_boundary/train/d-2_angle_tol_deg-10-5-2_corner_eps-2_min_sep-3_len_px-10-6_v5/final_vertices_heatmaps_sigma-3 \
        --sigma 3

    python ./data/generate_vertex_heatmaps.py \
        --json_dir ./data/deventer_512/train/junctions_unk_clean_merged \
        --save_dir ./data/deventer_512/train/junctions_unk_clean_merged_heatmaps_sigma-3 \
        --sigma 3

    # AI4SmallFarms
    python ./data/generate_vertex_heatmaps.py \
        --json_dir ../../agricultural_parcel_extraction/AI4SmallFarms_split_0.05/train_vertices \
        --save_dir ../../agricultural_parcel_extraction/AI4SmallFarms_split_0.05/train_vertex_heatmaps_sigma-3 \
        --sigma 3
    """
