#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate vertex heatmaps from COCO polygon annotations.

For each image, the script extracts polygon vertices from COCO segmentations
and generates a Gaussian heatmap centered on those vertices.

Usage example:
  python data/coco_to_vertex_heatmap.py \
    --ann_file /path/to/annotations.json \
    --image_dir /path/to/images \
    --save_dir /path/to/save_dir \
    --sigma 3
"""

import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
import matplotlib.pyplot as plt


def gaussian_2d(x, y, x0, y0, sigma):
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))


def generate_heatmap(vertex_locations, heatmap_shape, sigma=2.5):
    """
    Generate a vertex heatmap by taking max over per-vertex Gaussians.

    Args:
        vertex_locations: List or array of [x, y] pixel coordinates.
        heatmap_shape: Output shape as (H, W).
    """
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


def extract_vertices_from_coco(coco, img_id):
    """
    Extract all polygon vertices from COCO annotations of an image.
    Returns Nx2 array in (x, y). The closing point of each polygon is removed.
    """
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    vertices = []
    for ann in anns:
        segs = ann.get("segmentation", [])
        assert isinstance(segs, list), "RLE segmentation or other formats are not supported."

        for seg in segs:
            if len(seg) < 6:
                continue  # Fewer than 3 points.

            poly = np.asarray(seg, dtype=np.float32).reshape(-1, 2)

            # Many COCO polygons end with a duplicated first point; remove it if present.
            if poly.shape[0] >= 2 and np.allclose(poly[0], poly[-1]):
                poly = poly[:-1]

            vertices.append(poly)

    if len(vertices) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    return np.concatenate(vertices, axis=0)  # (N, 2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate vertex heatmaps from COCO polygon segmentations."
    )
    parser.add_argument("--ann_file", type=str, required=True, help="Path to COCO annotation JSON.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images referenced by COCO.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save heatmap .npy files.")
    parser.add_argument("--sigma", type=float, default=3.0, help="Gaussian sigma for heatmap generation.")
    parser.add_argument("--ext", type=str, default=".npy", help="Output file extension (.npy recommended).")

    # Debug options
    parser.add_argument("--save_vis", action="store_true", help="Save visualization PNGs (slow).")
    parser.add_argument("--vis_cmap", type=str, default="jet", help="Colormap for visualization.")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_vis:
        os.makedirs(os.path.join(args.save_dir, "vis"), exist_ok=True)

    coco = COCO(args.ann_file)
    img_ids = coco.getImgIds()

    for img_id in tqdm(img_ids, desc="Generating vertex heatmaps"):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        stem = os.path.splitext(os.path.basename(file_name))[0]

        img_path = os.path.join(args.image_dir, file_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        H, W = img.shape[:2]

        # Extract vertices in float32 (x, y) format.
        vertices = extract_vertices_from_coco(coco, img_id)

        # Clip to image bounds (x -> W, y -> H), then round to integer pixels.
        if vertices.shape[0] > 0:
            vertices[:, 0] = np.clip(vertices[:, 0], 0, W - 1)  # x
            vertices[:, 1] = np.clip(vertices[:, 1], 0, H - 1)  # y
            vertices = np.round(vertices).astype(np.int32)      # integer pixels
        else:
            vertices = vertices.astype(np.int32)

        # Generate the heatmap at the original image resolution.
        heatmap = generate_heatmap(vertices, (H, W), sigma=args.sigma)

        # Save the heatmap.
        save_path = os.path.join(args.save_dir, f"{stem}{args.ext}")
        if args.ext.lower() == ".npy":
            np.save(save_path, heatmap.astype(np.float32))
        else:
            # Fallback: save as 8-bit image if user insists (not recommended)
            cv2.imwrite(save_path, (heatmap * 255).astype(np.uint8))

        # Optional visualization.
        if args.save_vis:
            vis_path = os.path.join(args.save_dir, "vis", f"{stem}.png")
            plt.figure(figsize=(6, 6))
            plt.imshow(heatmap, cmap=args.vis_cmap)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(vis_path, dpi=200)
            plt.close()


if __name__ == "__main__":
    main()
