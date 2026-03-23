#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Betti errors (beta0 error and beta1 error) between COCO-format
polygon predictions and ground-truth annotations for a (single) class.

For each image, a binary mask is constructed and the following quantities are computed:
- β_0: the number of connected components, using 8-connectivity on the foreground;
- β_1: the number of holes, defined as background connected components (4-connectivity) that do not touch the image boundary;
- Betti error 0: |β_0^pred - β_0^gt|
- Betti error 1: |β_1^pred - β_1^gt|

The script reports both the overall (macro-averaged) metrics and per-image results in JSON format.

Assumptions:
- Both GT and Pred are COCO-style JSONs.
- Each instance polygon is stored in "segmentation" as a list of polygons:
  [[x1,y1,x2,y2,...], [hole1_x1,hole1_y1,...], ...]
  where the first polygon is the outer boundary, the rest are holes.
- The prediction JSON may contain multiple instances per image_id.
- We build masks per image_id using GT "images" metadata for image size.

Outputs:
- Prints dataset-level mean Betti error 0/1.
- Writes per-image results to a JSON file in --output path.
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import cv2
from skimage.measure import label

def load_coco(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def build_image_size_map(gt_data):
    """
    Return dict: image_id -> (height, width, file_name)
    """
    img_map = {}
    for img in gt_data.get('images', []):
        img_map[img['id']] = (img['height'], img['width'], img.get('file_name', str(img['id'])))
    return img_map

def group_segmentations_by_image(coco_like, score_thresh=0.0):
    """
    Return image_id -> list of segmentations.

    Each segmentation is stored as [outer, hole1, ...]. For prediction files,
    instances with score < score_thresh are skipped when a score is provided.
    """
    segs_by_image = defaultdict(list)
    for ann in coco_like.get('annotations', coco_like if isinstance(coco_like, list) else []):
        if isinstance(coco_like, dict):
            image_id = ann['image_id']
            seg = ann.get('segmentation', [])
            segs_by_image[image_id].append(seg)
        else:
            image_id = ann['image_id']
            if 'score' in ann and ann['score'] < score_thresh:
                continue
            seg = ann.get('segmentation', [])
            segs_by_image[image_id].append(seg)
    return segs_by_image

def polygons_to_mask(H, W, segmentations):
    """
    Rasterize a list of instance segmentations into a single binary mask.
    - For each instance:
        * Fill outer polygon with 1
        * Carve out its holes with 0
    - Instances are composited by OR (outer) and carving their own holes afterwards.
    """
    mask = np.zeros((H, W), dtype=np.uint8)

    for seg in segmentations:
        if not seg:
            continue
        # Fill the outer ring first.
        outer = np.array(seg[0], dtype=np.float32).reshape(-1, 2)
        outer = np.round(outer).astype(np.int32)
        cv2.fillPoly(mask, [outer], 1)

        # Then carve hole rings.
        if len(seg) > 1:
            for hole in seg[1:]:
                hole_arr = np.array(hole, dtype=np.float32).reshape(-1, 2)
                hole_arr = np.round(hole_arr).astype(np.int32)
                cv2.fillPoly(mask, [hole_arr], 0)

    return mask.astype(bool)

def betti_numbers_from_mask(mask_bool, fg_conn=2, bg_conn=1):
    """
    Compute Betti numbers on a binary mask.

    Conventions:
      - Foreground uses 8-connectivity by default.
      - Background uses 4-connectivity by default.
      - beta0 is the number of foreground connected components.
      - beta1 is the number of background components that do not touch the image border.

    Returns: (beta0, beta1) as integers
    """
    if mask_bool.size == 0:
        return 0, 0

    # beta0: foreground connected components.
    _, n_fg = label(mask_bool.astype(np.uint8), connectivity=fg_conn, return_num=True)

    # beta1: background connected components that do not touch the border.
    bg = (~mask_bool).astype(np.uint8)
    labeled_bg = label(bg, connectivity=bg_conn)
    bg_labels = labeled_bg

    H, W = mask_bool.shape
    # Collect background labels that touch the image border.
    border_mask = np.zeros_like(mask_bool, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    border_labels = set(np.unique(bg_labels[border_mask]))

    # Collect all background component labels except 0.
    all_bg_labels = set(np.unique(bg_labels))
    if 0 in all_bg_labels:
        all_bg_labels.remove(0)

    # Holes are background components that do not touch the border.
    hole_labels = [lab for lab in all_bg_labels if lab not in border_labels]
    beta1 = len(hole_labels)

    beta0 = int(n_fg)
    beta1 = int(beta1)
    return beta0, beta1

def main():
    parser = argparse.ArgumentParser(
        description="Compute Betti-number errors between COCO GT and prediction polygons."
    )
    parser.add_argument("--gt_file", required=True, help="Path to COCO GT JSON (must contain images & annotations).")
    parser.add_argument("--dt_file", required=True, help="Path to COCO prediction JSON (COCO results: list or dict with annotations).")
    parser.add_argument("--score_thresh", type=float, default=0.0, help="Score threshold for predictions (if pred has 'score').")
    parser.add_argument("--fg_conn", type=int, default=2, choices=[1,2], help="Foreground connectivity: 1=4-connected, 2=8-connected (default 2).")
    parser.add_argument("--bg_conn", type=int, default=1, choices=[1,2], help="Background connectivity: 1=4-connected (default), 2=8-connected.")
    args = parser.parse_args()

    gt_data = load_coco(args.gt_file)

    # The prediction file can be a COCO results list or a dict with annotations.
    with open(args.dt_file, 'r') as f:
        pred_raw = json.load(f)
    if isinstance(pred_raw, dict):
        pred_data = pred_raw
    else:
        # Wrap list predictions for unified downstream handling.
        pred_data = {"annotations": pred_raw}

    img_map = build_image_size_map(gt_data)
    gt_segs = group_segmentations_by_image(gt_data, score_thresh=0.0)
    pred_segs = group_segmentations_by_image(pred_data, score_thresh=args.score_thresh)

    per_image = {}
    chi_err_list, e_list, e0_list, e1_list = [], [], [], []

    # Evaluate over the GT image set.
    for image_id, (H, W, fname) in img_map.items():
        segs_gt = gt_segs.get(image_id, [])
        segs_pred = pred_segs.get(image_id, [])

        # Build binary masks.
        mask_gt = polygons_to_mask(H, W, segs_gt)
        mask_pred = polygons_to_mask(H, W, segs_pred)

        # Compute Betti numbers.
        b0_gt, b1_gt = betti_numbers_from_mask(mask_gt, fg_conn=args.fg_conn, bg_conn=args.bg_conn)
        b0_pr, b1_pr = betti_numbers_from_mask(mask_pred, fg_conn=args.fg_conn, bg_conn=args.bg_conn)

        # Compute per-image errors.
        e0 = abs(b0_pr - b0_gt)
        e1 = abs(b1_pr - b1_gt)
        e = e0 + e1

        chi_gt = b0_gt - b1_gt
        chi_pr = b0_pr - b1_pr
        chi_err = abs(chi_pr - chi_gt)

        per_image[str(image_id)] = {
            "file_name": fname,
            "beta0_gt": int(b0_gt),
            "beta1_gt": int(b1_gt),
            "beta0_pred": int(b0_pr),
            "beta1_pred": int(b1_pr),
            "betti_error_0": int(e0),
            "betti_error_1": int(e1),
        }
        chi_err_list.append(chi_err)
        e_list.append(e)
        e0_list.append(e0)
        e1_list.append(e1)

    mean_chi_err = float(np.mean(chi_err_list)) if chi_err_list else 0.0
    mean_e = float(np.mean(e_list)) if e_list else 0.0
    mean_e0 = float(np.mean(e0_list)) if e0_list else 0.0
    mean_e1 = float(np.mean(e1_list)) if e1_list else 0.0

    summary = {
        "mean_chi_err": mean_chi_err,
        "mean_betti_error": mean_e,
        "mean_betti_error_0": mean_e0,
        "mean_betti_error_1": mean_e1,
        "num_images": len(per_image),
        "connectivity": {
            "foreground": "8-connected" if args.fg_conn == 2 else "4-connected",
            "background": "4-connected" if args.bg_conn == 1 else "8-connected"
        },
        "note": "beta0: #foreground components; beta1: #holes (background components not touching border)."
    }

    out = {
        "summary": summary,
        "per_image": per_image
    }

    # Saving can be re-enabled later if detailed JSON output is needed.

    print("== Betti Errors (macro average) ==")
    print(f"Mean chi error: {mean_chi_err:.2f}")
    print(f"Mean Betti error: {mean_e:.2f}")
    print(f"Mean Betti error 0: {mean_e0:.2f}")
    print(f"Mean Betti error 1: {mean_e1:.2f}")
    print(f"Images evaluated:   {len(per_image)}")

if __name__ == "__main__":
    main()
