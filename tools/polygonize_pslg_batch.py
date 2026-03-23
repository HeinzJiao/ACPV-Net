#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch polygonize semantic label maps with the PSLG-based post-processing pipeline.

This script runs the main multi-category polygonization workflow or an ablation
baseline on a folder of predicted label maps, saves per-image polygon JSON
files, exports per-category COCO-style predictions, and renders overview
visualizations.

Example:
    python tools/polygonize_pslg_batch.py \
        --seg_dir /path/to/seg_npy \
        --junction_dir /path/to/vertex_json \
        --categories 0 1 2 3 4 \
        --cats_out_dir /path/to/coco_json \
        --mode ours --vss --dp_fix
"""

import os
import cv2
import json
import argparse
import numpy as np
from polygonize_pslg_one_image import (
    process_one_image_categories, flatten_to_xylist
)
from shapely.geometry import Polygon
from tqdm import tqdm


ID2NAME = {
    0: "building",
    1: "road_bridge",
    2: "unvegetated",
    3: "vegetation",
    4: "water",
}
def _bbox_from_xy(ext_xy):
    xs = [p[0] for p in ext_xy]; ys = [p[1] for p in ext_xy]
    x0, y0 = float(min(xs)), float(min(ys))
    x1, y1 = float(max(xs)), float(max(ys))
    return [x0, y0, x1 - x0, y1 - y0]


def area_bbox_with_holes(ext_xy, holes_xy):
    """
    Compute polygon area and bounding box from one exterior ring and its holes.

    The bounding box is always measured from the exterior ring and returned as
    `[x_min, y_min, width, height]`.
    """
    # Use the exterior ring for the bounding box.
    bbox = _bbox_from_xy(ext_xy) if len(ext_xy) >= 3 else [0.0, 0.0, 0.0, 0.0]

    holes_valid = [h for h in holes_xy if len(h) >= 3]
    poly = Polygon(ext_xy, holes=holes_valid)
    if not poly.is_valid:
        # Standard fix for self-intersections, duplicate points, or tiny gaps.
        poly = poly.buffer(0)
    return float(max(poly.area, 0.0)), [float(b) for b in bbox]


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("1","true","t","yes","y")


def _flatten_xy(cnt):
        """Convert an `Nx1x2` or `Nx2` contour to `[x1, y1, x2, y2, ...]`."""
        c = np.squeeze(cnt, axis=1) if cnt.ndim == 3 else cnt
        if c.ndim != 2 or c.shape[1] != 2:
            return []
        return [float(v) for xy in c for v in xy]

def _approx(cnt, eps):
    """Apply DP simplification to a contour while keeping at least three points."""
    if cnt is None or len(cnt) == 0:
        return None
    approx = cv2.approxPolyDP(cnt, epsilon=eps, closed=True)
    # Reject degenerate contours caused by repeated points.
    if approx is None or len(approx) < 3:
        return None
    return approx


# Ablation baseline: contour extraction plus DP, without PSLG reconstruction.
def polygonize_per_class_no_pslg_dp(
    seg_path: str,
    categories,
    vis_dir: str = None,
    vis_scale: int = 1,
    dp_epsilon: float = 2.5,
):
    """
    No-PSLG baseline:
      - Extract contours directly from each category mask.
      - Simplify them in pixel space with Douglas-Peucker.
      - Pack one exterior ring and optional holes as
        `[ext_flat, hole1_flat, ...]`.
      - Return the same category-indexed structure as the main pipeline.

    Notes:
      - Predicted-junction and VSS-related arguments are intentionally unused in
        this ablation, but the interface stays compatible.
      - `dp_epsilon` is measured in pixels.
    """
    # Load the multi-class label map.
    L = np.load(seg_path).astype(np.int32)  # (H, W)
    H, W = L.shape

    # Create the optional visualization directory.
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    # Match the output structure of the main polygonization pipeline.
    results = {int(c): [] for c in categories}

    # Use hierarchy-aware contour extraction so exteriors can recover holes.
    RETR_MODE = cv2.RETR_TREE
    CHAIN_MODE = cv2.CHAIN_APPROX_NONE  # Keep dense contour samples for DP.

    for cat in categories:
        cat = int(cat)
        # Extract the binary mask for the current category.
        mask_bin = (L == cat).astype(np.uint8)  # 0/1
        if mask_bin.max() == 0:
            continue

        # Extract contours and the parent-child hierarchy.
        contours, hierarchy = cv2.findContours(mask_bin, RETR_MODE, CHAIN_MODE)
        if not contours or hierarchy is None:
            continue
        hierarchy = hierarchy[0]  # (N,4): [next, prev, first_child, parent]

        # Iterate over exterior contours only (`parent == -1`).
        for i, h in enumerate(hierarchy):
            parent = h[3]
            if parent != -1:
                continue

            ext = contours[i]
            if ext is None or len(ext) < 3:
                continue

            # Simplify the exterior contour.
            ext_dp = _approx(ext, dp_epsilon)
            if ext_dp is None:
                continue

            # Collect direct child contours as holes.
            holes_dp = []
            child = h[2]
            while child != -1:
                hole = contours[child]
                if hole is not None and len(hole) >= 3:
                    hole_dp = _approx(hole, dp_epsilon)
                    if hole_dp is not None:
                        holes_dp.append(hole_dp)
                child = hierarchy[child][0]  # Move to the next sibling hole.

            # Pack the exterior ring and holes as flattened float lists.
            seg = []
            seg.append(_flatten_xy(ext_dp))
            for hdp in holes_dp:
                seg.append(_flatten_xy(hdp))

            # Skip invalid or empty polygons.
            if len(seg[0]) < 6:  # At least three vertices.
                continue

            results[cat].append(seg)

        # Optional per-category contour visualization.
        if vis_dir:
            vis = np.zeros((H * vis_scale, W * vis_scale, 3), dtype=np.uint8)
            color = (0, 255, 255)
            for seg in results[cat]:
                # Exterior ring.
                ext = np.array(seg[0], dtype=np.float32).reshape(-1, 2)
                ext_i = np.round(ext * vis_scale).astype(np.int32)
                cv2.polylines(vis, [ext_i], isClosed=True, color=color, thickness=1)
                # Hole rings.
                for hole in seg[1:]:
                    h = np.array(hole, dtype=np.float32).reshape(-1, 2)
                    h_i = np.round(h * vis_scale).astype(np.int32)
                    cv2.polylines(vis, [h_i], isClosed=True, color=(0, 128, 255), thickness=1)

            cv2.imwrite(os.path.join(vis_dir, f"cat_{cat}.png"), vis)

    return results


def instance_score_mean_prob(prob_map, cat_id, ext_xy, holes_xy, H, W):
    """
    prob_map: (C, H, W) float32 in [0,1]
    cat_id: int
    ext_xy: [(x,y),...]
    holes_xy: list of [(x,y),...]
    """
    if prob_map is None:
        return 1.0
    if prob_map.ndim != 3 or cat_id < 0 or cat_id >= prob_map.shape[0]:
        return 1.0

    p = prob_map[cat_id]  # (H, W)

    # exterior mask
    mask_ext = np.zeros((H, W), dtype=np.uint8)
    poly_ext = np.array(ext_xy, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask_ext, [poly_ext], 1)

    # holes mask
    if holes_xy:
        mask_hole = np.zeros((H, W), dtype=np.uint8)
        for h in holes_xy:
            if len(h) < 3:
                continue
            poly_h = np.array(h, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask_hole, [poly_h], 1)
        mask = (mask_ext == 1) & (mask_hole == 0)
    else:
        mask = (mask_ext == 1)

    if not np.any(mask):
        return 0.0

    return float(np.mean(p[mask]))


def main():
    parser = argparse.ArgumentParser(
        description="Batch polygonization: ours (PSLG+VSS) or ablation (no PSLG, pure DP per region)."
    )

    # ------------------------------------------------------------------
    # Required I/O
    # ------------------------------------------------------------------
    parser.add_argument("--seg_dir", required=True,
                        help="Directory of predicted label maps (.npy).")
    parser.add_argument("--junction_dir", required=True,
                        help="Directory of predicted vertices JSONs: [[x,y], ...] (matched by basename).")
    parser.add_argument("--categories", type=int, nargs="+", required=True,
                        help="Target category IDs, e.g. --categories 0 1 2")
    parser.add_argument("--cats_out_dir", required=True,
                        help="Directory to save per-category COCO prediction JSONs.")
    parser.add_argument("--prob_dir", default=None,
                    help="Directory of predicted probability maps (.npy), shape (C,H,W). Matched by basename.")


    # ------------------------------------------------------------------
    # Run mode: ours vs ablation
    # ------------------------------------------------------------------
    parser.add_argument("--mode", type=str, default="ours",
                        choices=["ours", "ablation_dp"],
                        help=("Run mode: "
                              "'ours' = PSLG + VSS (+ optional DP-fix); "
                              "'ablation_dp' = no PSLG, pure DP per semantic region."))

    # ------------------------------------------------------------------
    # Optional visualization
    # ------------------------------------------------------------------
    parser.add_argument("--vis_dir", default=None,
                        help="Optional root dir for per-image step visualizations: <vis_dir>/<name>/...")
    parser.add_argument("--vis_scale", type=int, default=1,
                        help="Visualization scale factor (default: 1).")

    # ------------------------------------------------------------------
    # Core geometric hyper-parameters (ours-mode)
    # ------------------------------------------------------------------
    parser.add_argument("--dist_thresh", type=float, default=5.0,
                        help="Snap radius for predicted vertices to boundary polylines (pixels).")
    parser.add_argument("--corner_eps", type=float, default=2.0,
                        help="Radius to treat a point as too close to a structural corner (pixels).")

    # ------------------------------------------------------------------
    # Point selection strategy (ours-mode)
    # ------------------------------------------------------------------
    parser.add_argument("--vss", action="store_true",
                        help="Enable vertex-guided subset selection (VSS). "
                             "If not set, pure DP simplification is used.")
    parser.add_argument("--dp_fix", action="store_true",
                        help="Enable DP-based safeguard (effective only when --vss is set).")
    parser.add_argument("--dp_epsilon_dp", type=float, default=1.0,
                        help="RDP epsilon for pure DP fallback (used when vss is OFF in ours-mode)."
    )
    parser.add_argument("--dp_epsilon_fix", type=float, default=2.5,
                        help="RDP epsilon for DP safeguard in VSS mode (used when dp_fix is ON).")

    # ------------------------------------------------------------------
    # Ablation hyper-parameters (ablation_dp-mode)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--ablation_dp_epsilon", type=float, default=2.5,
        help="RDP epsilon for ablation mode (no PSLG, pure DP per region)."
    )

    args = parser.parse_args()

    # Prepare output dirs
    os.makedirs(args.cats_out_dir, exist_ok=True)
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    # COCO predictions per category
    per_cat_preds = {int(c): [] for c in args.categories}

    # ------------------------------------------------------------------
    # Iterate inputs
    # ------------------------------------------------------------------
    seg_files = sorted([f for f in os.listdir(args.seg_dir) if f.endswith('.npy')])

    for f in tqdm(seg_files, desc=f"Polygonizing ({args.mode})"):
        name = os.path.splitext(f)[0]
        seg_path  = os.path.join(args.seg_dir, f)
        junc_path = os.path.join(args.junction_dir, name + '.json')

        prob_map = None
        if args.prob_dir is not None:
            prob_path = os.path.join(args.prob_dir, name + ".npy")
            if os.path.exists(prob_path):
                prob_map = np.load(prob_path).astype(np.float32)

        # If junction is missing:
        # - ours-mode: skip
        # - ablation: can run without junctions
        if args.mode == "ours" and (not os.path.exists(junc_path)):
            continue

        vis_subdir = os.path.join(args.vis_dir, name) if args.vis_dir else None

        # --------------------------------------------------------------
        # Run polygonization
        # --------------------------------------------------------------
        if args.mode == "ours":
            results = process_one_image_categories(
                seg_path=seg_path,
                junction_json=junc_path,
                categories=args.categories,
                dist_thresh=args.dist_thresh,
                corner_eps=args.corner_eps,
                vis_dir=vis_subdir,
                vis_scale=args.vis_scale,
                vss=args.vss,
                dp_fix=args.dp_fix,
                dp_epsilon_dp=args.dp_epsilon_dp,
                dp_epsilon_fix=args.dp_epsilon_fix,
            )
        else:
            results = polygonize_per_class_no_pslg_dp(
                seg_path=seg_path,
                categories=args.categories,
                vis_dir=vis_subdir,
                vis_scale=args.vis_scale,
                dp_epsilon=args.ablation_dp_epsilon,
            )

        # --------------------------------------------------------------
        # COCO predictions (per-category)
        # --------------------------------------------------------------
        L = np.load(seg_path).astype(np.int32)
        H, W = L.shape
        image_id = int(name.split('_')[-1])

        for cat in args.categories:
            cat = int(cat)
            inst_list = results.get(cat, [])
            for inst in inst_list:
                if not inst:
                    continue
                ext_xy = flatten_to_xylist(inst[0])
                holes_xy = [flatten_to_xylist(hv) for hv in inst[1:]]
                area, bbox = area_bbox_with_holes(ext_xy, holes_xy)

                score = instance_score_mean_prob(
                    prob_map=prob_map,
                    cat_id=cat,
                    ext_xy=ext_xy,
                    holes_xy=holes_xy,
                    H=H,
                    W=W
                )

                coco_obj = {
                    "image_id": image_id,
                    "category_id": 100,
                    "segmentation": inst,  # [[x1,y1,...], [hole...], ...]
                    "score": score,
                    "iscrowd": 0,
                    "area": area,
                    "bbox": bbox,
                }
                per_cat_preds[cat].append(coco_obj)

    # ------------------------------------------------------------------
    # Save COCO prediction json per category
    # ------------------------------------------------------------------
    for cat in args.categories:
        cat = int(cat)
        cat_name = ID2NAME.get(cat, f"cat_{cat}")
        save_path = os.path.join(args.cats_out_dir, f"{cat_name}.json")
        with open(save_path, 'w') as fp:
            json.dump(per_cat_preds[cat], fp)

if __name__ == "__main__":
    main()

