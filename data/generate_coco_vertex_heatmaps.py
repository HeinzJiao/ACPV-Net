#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge COCO polygon vertices across classes and save per-image heatmaps.

Example:
    python tools/generate_coco_vertex_heatmaps.py \
        --annotation-path /path/to/annotations --output-dir /path/to/heatmaps
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def collect_vertices(annotation_path):
    paths = sorted(annotation_path.glob("*.json")) if annotation_path.is_dir() else [annotation_path]
    if not paths:
        raise ValueError(f"No annotation JSON files found: {annotation_path}")
    images, vertices = {}, {}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            coco = json.load(stream)
        image_names = {}
        for image in coco["images"]:
            name = Path(image["file_name"]).stem
            metadata = (image["file_name"], int(image["height"]), int(image["width"]))
            if name in images and images[name] != metadata:
                raise ValueError(f"Conflicting image names or dimensions: {name} in {path}")
            images[name] = metadata
            vertices.setdefault(name, set())
            image_names[image["id"]] = name
        for annotation in coco["annotations"]:
            name = image_names[annotation["image_id"]]
            _, height, width = images[name]
            segmentation = annotation.get("segmentation", [])
            if not isinstance(segmentation, list):
                raise ValueError(f"Polygon segmentation required; RLE has no explicit vertices: {path}")
            for ring in segmentation:
                points = np.asarray(ring, dtype=np.float64)
                if points.ndim != 1 or points.size % 2 or not np.isfinite(points).all():
                    raise ValueError(f"Invalid polygon coordinates in annotation {annotation['id']}: {path}")
                points = points.reshape(-1, 2)
                # Match the existing vertex generator's image-bound clipping.
                points[:, 0] = np.clip(points[:, 0], 0, width - 1)
                points[:, 1] = np.clip(points[:, 1], 0, height - 1)
                vertices[name].update(map(tuple, points.tolist()))
        print(f"Loaded {path.name}: {len(coco['images'])} images, {len(coco['annotations'])} annotations", flush=True)
    return images, vertices


def generate_heatmap(vertices, shape, sigma):
    heatmap = np.zeros(shape, dtype=np.float64)
    # This window includes all Gaussian values retained by the 1e-8 cutoff.
    radius = math.ceil(sigma * math.sqrt(-2 * math.log(1e-8))) + 1
    for x, y in sorted(vertices):
        left, right = max(0, math.floor(x) - radius), min(shape[1], math.ceil(x) + radius + 1)
        top, bottom = max(0, math.floor(y) - radius), min(shape[0], math.ceil(y) + radius + 1)
        dx = np.arange(left, right, dtype=np.float64) - x
        dy = np.arange(top, bottom, dtype=np.float64) - y
        gaussian = np.exp(-(dy[:, None] ** 2 + dx[None, :] ** 2) / (2 * sigma ** 2))
        patch = heatmap[top:bottom, left:right]
        np.maximum(patch, gaussian, out=patch)
    peak = heatmap.max()
    if peak > 0:
        heatmap /= peak
    heatmap[heatmap < 1e-8] = 0
    return heatmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-path", type=Path, required=True,
                        help="One COCO JSON file or a directory of per-class COCO JSON files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination for per-image heatmaps.")
    parser.add_argument("--sigma", type=float, default=3.0, help="Gaussian standard deviation in pixels.")
    parser.add_argument("--format", choices=("npy", "pt"), default="npy", help="Output file format.")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32", help="Saved array precision.")
    args = parser.parse_args()
    if not math.isfinite(args.sigma) or args.sigma <= 0:
        parser.error("--sigma must be finite and positive.")
    if args.annotation_path.resolve() == args.output_dir.resolve():
        parser.error("Use a separate output directory for heatmaps.")
    if args.format == "pt":
        import torch
    images, vertices = collect_vertices(args.annotation_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_vertices, empty_images = 0, 0
    for index, name in enumerate(sorted(images), 1):
        _, height, width = images[name]
        heatmap = generate_heatmap(vertices[name], (height, width), args.sigma).astype(args.dtype)
        destination = args.output_dir / f"{name}.{args.format}"
        if args.format == "pt":
            torch.save(torch.from_numpy(heatmap), destination)
        else:
            np.save(destination, heatmap)
        total_vertices += len(vertices[name])
        empty_images += int(not vertices[name])
        if index % 100 == 0 or index == len(images):
            print(f"Saved {index}/{len(images)} heatmaps", flush=True)
    print(f"Unique vertices: {total_vertices}; images without vertices: {empty_images}")
    print(f"Output: {args.output_dir} ({args.format}, {args.dtype}, sigma={args.sigma})")


if __name__ == "__main__":
    main()

    """
    python tools/generate_coco_vertex_heatmaps.py \
    --annotation-path "$HOME/ACPV-Net/data/deventer_512/test/annotations" \
    --output-dir "$HOME/ACPV-Net/data/deventer_512/test/vertex_heatmaps" \
    --sigma 3 \
    --dtype float32
    
    """
