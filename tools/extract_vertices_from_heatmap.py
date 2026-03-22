"""
Extract vertices from predicted heatmaps and evaluate vertex-level precision/recall (TP/FP/FN).
"""

import os
import numpy as np
import json
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

def parse_args():
    parser = argparse.ArgumentParser(description="Extract predicted vertices from .npy heatmaps and evaluate.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with predicted vertex heatmaps (.npy)')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save extracted vertices (.json)')
    parser.add_argument('--threshold', type=float, default=0.1, help='Pixel-wise threshold for vertex detection')
    parser.add_argument('--topk', type=int, default=300, help='Maximum number of vertices per image')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size used in NMS')
    parser.add_argument('--gt_dir', type=str, default=None, help='Optional: ground truth vertex .json directory')
    parser.add_argument('--dist_thresh', type=float, default=5.0, help='Matching distance for evaluation')
    return parser.parse_args()


def non_maximum_suppression(a, kernel_size=3):
    """
    Apply local non-maximum suppression (NMS) to a heatmap tensor.

    Args:
        a (Tensor): Input heatmap of shape [N, C, H, W].
        kernel_size (int): Window size for max pooling (must be odd).

    Returns:
        Tensor: Tensor with the same shape as `a`, where only local maxima
        are preserved and all other locations are set to zero.
    """
    assert kernel_size % 2 == 1, "kernel_size must be an odd integer."
    kernel_size = int(kernel_size)
    pad = kernel_size // 2

    # Local max pooling
    ap = F.max_pool2d(a, kernel_size, stride=1, padding=pad)

    # Keep only positions equal to the local maximum
    mask = (a == ap).float()
    return a * mask


def extract_vertices(prob_map: np.ndarray, threshold: float, topk: int, kernel_size: int):
    """
    Extract vertices via NMS + top-k + thresholding.

    Args:
        prob_map: heatmap array in shape (H, W) or (3, H, W)
        threshold: keep vertices with score > threshold
        topk: max number of candidates kept after sorting
        kernel_size: NMS window size (odd)

    Returns:
        List[[x, y]]: integer pixel coordinates.
    """
    if prob_map.ndim == 3 and prob_map.shape[0] == 3:
        prob_map = prob_map.mean(axis=0)  # (H, W)
        prob_map = (prob_map + 1.0) / 2.0  # [-1, 1] → [0, 1]
    elif prob_map.ndim != 2:
        raise ValueError(f"Unsupported shape {prob_map.shape}, expected (H, W) or (3, H, W)")

    tensor_map = torch.from_numpy(prob_map).unsqueeze(0).float()  # (1, H, W)
    nms_map = non_maximum_suppression(tensor_map, kernel_size=kernel_size).squeeze(0)  # (H, W)

    flat = nms_map.flatten()
    scores, indices = torch.topk(flat, k=topk)
    keep = scores > threshold

    indices = indices[keep]
    y = indices // prob_map.shape[1]
    x = indices % prob_map.shape[1]

    vertices = [[int(x_), int(y_)] for x_, y_ in zip(x.tolist(), y.tolist())]
    return vertices


def load_gt_vertices(json_path):
    """
    Load GT vertices from a JSON file.

    Expected format: list of [x, y, ...] or [x, y].
    Returns: List[[x, y]] as integers.
    """
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r') as f:
        return [[int(x), int(y)] for x, y, *_ in json.load(f)]


def match_vertices(preds, gts, dist_thresh):
    """
    One-to-one matching between predicted and GT vertices using nearest neighbor search.

    Returns:
        tp, fp, fn
    """
    if len(preds) == 0 or len(gts) == 0:
        return 0, len(preds), len(gts)

    tree = cKDTree(np.array(gts))
    matched_gt = set()
    tp = 0

    for p in preds:
        dist, idx = tree.query(p, distance_upper_bound=dist_thresh)
        if dist != np.inf and idx not in matched_gt:
            tp += 1
            matched_gt.add(idx)

    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    filenames = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".npy")])
    all_tp, all_fp, all_fn = 0, 0, 0

    for fname in tqdm(filenames, desc="Processing"):
        name = os.path.splitext(fname)[0]
        path = os.path.join(args.input_dir, fname)
        prob_map = np.load(path)

        # 1. Extract vertices
        vertices = extract_vertices(prob_map, threshold=args.threshold, topk=args.topk, kernel_size=args.kernel_size)

        # 2. Save to JSON
        with open(os.path.join(args.save_dir, f"{name}.json"), 'w') as f:
            json.dump(vertices, f)

        # 3. Evaluate (if GT provided)
        if args.gt_dir:
            gt_path = os.path.join(args.gt_dir, f"{name}.json")
            gt_vertices = load_gt_vertices(gt_path)
            tp, fp, fn = match_vertices(vertices, gt_vertices, dist_thresh=args.dist_thresh)
            all_tp += tp
            all_fp += fp
            all_fn += fn

    if args.gt_dir:
        precision = all_tp / (all_tp + all_fp + 1e-8)
        recall = all_tp / (all_tp + all_fn + 1e-8)
        print("\n=== Evaluation Summary ===")
        print(f"TP: {all_tp} | FP: {all_fp} | FN: {all_fn}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")


if __name__ == "__main__":
    main()
