#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate polygon predictions against GT multi-class segmentation masks
(named by image file_name, e.g., True_Ortho_2049_4713_0_0.png).

Metrics (6, APLS excluded):
- Pixel-wise mask metrics:
    PAmask, F1mask, mIoUmask
- Topology-wise metrics (boundary-based, dilated by δ):
    PAtopo, F1topo, mIoUtopo

Averaging strategy (standard & recommended):
- Accumulate TP / FP / FN / TN over the whole dataset PER CLASS
- Compute per-class metrics
- Macro-average over classes

Notes:
- delta (δ) is the topo dilation radius (TopDiG protocol).
- line_thickness is only for polygon->pixel rasterization of boundaries.
"""

import os
import json
import cv2
import numpy as np
from collections import defaultdict

# Class name to label-id mapping.
NAME2ID = {
    "artificial_structure": 0,
    "building": 1,
    "road_bridge": 2,
    "unvegetated": 3,
    "vegetation": 4,
    "water": 5,
    "unknown": 6,
}

def infer_cat_id_from_stem(stem: str):
    s = stem.lower()
    matches = [(len(k), NAME2ID[k]) for k in NAME2ID if k in s]
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]

# Geometry helpers.
def flatten_to_xylist(flat):
    a = np.asarray(flat, dtype=np.float32).reshape(-1, 2)
    a = np.rint(a).astype(np.int32)
    return a

def clip_poly(poly, W, H):
    poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)
    return poly

# Load predictions and image_id -> file_name metadata when available.
def load_per_image_instances(pred_dir):
    """
    Returns:
      per_image_per_class[image_id][class_key] = list of segmentations
      id2name_pred[image_id] = file_name if the prediction JSON provides an images table
    """
    per_image_per_class = defaultdict(lambda: defaultdict(list))
    id2name_pred = {}

    for fname in sorted(os.listdir(pred_dir)):
        if not fname.lower().endswith(".json"):
            continue
        class_key = os.path.splitext(fname)[0]
        path = os.path.join(pred_dir, fname)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # COCO dict case.
        if isinstance(data, dict):
            if "images" in data and isinstance(data["images"], list):
                for im in data["images"]:
                    try:
                        iid = int(im["id"])
                        fn = im.get("file_name", None)
                        if fn is not None and iid not in id2name_pred:
                            id2name_pred[iid] = os.path.basename(fn)
                    except Exception:
                        pass

            if "annotations" in data:
                data = data["annotations"]
            else:
                raise ValueError(f"{fname} is a dict but missing 'annotations' field.")

        # Plain annotation-list case.
        if not isinstance(data, list):
            raise ValueError(f"{fname} must be list or COCO dict.")

        for ann in data:
            image_id = int(ann["image_id"])
            seg = ann.get("segmentation", [])
            if not isinstance(seg, list) or len(seg) == 0:
                continue
            per_image_per_class[image_id][class_key].append(seg)

    return per_image_per_class, id2name_pred

# Load image_id -> file_name from a COCO GT file or folder.
def load_id2name_from_coco_path(coco_path):
    """
    coco_path can be:
      - a COCO JSON file
      - a folder containing multiple COCO JSON files
    All discovered images tables are merged.
    """
    id2name = {}

    def _load_one_json(jpath):
        nonlocal id2name
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                d = json.load(f)
            imgs = d.get("images", [])
            if not isinstance(imgs, list):
                return
            for im in imgs:
                try:
                    iid = int(im["id"])
                    fn = im.get("file_name", None)
                    if fn is None:
                        continue
                    fn = os.path.basename(fn)
                    if iid not in id2name:
                        id2name[iid] = fn
                except Exception:
                    pass
        except Exception:
            pass

    if coco_path is None:
        return id2name

    if os.path.isdir(coco_path):
        for fname in sorted(os.listdir(coco_path)):
            if fname.lower().endswith(".json"):
                _load_one_json(os.path.join(coco_path, fname))
    else:
        _load_one_json(coco_path)

    return id2name

# GT loading.
def load_gt_mask(gt_path):
    if gt_path.lower().endswith(".npy"):
        return np.load(gt_path).astype(np.int32)
    img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(gt_path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.int32)

# Rasterization.
def rasterize_instance_mask(seg, H, W):
    """Interior mask: fill ext then subtract holes."""
    m = np.zeros((H, W), dtype=np.uint8)

    ext = flatten_to_xylist(seg[0])
    if ext.shape[0] < 3:
        return m
    ext = clip_poly(ext, W, H)
    cv2.fillPoly(m, [ext.reshape(-1, 1, 2)], 1, lineType=cv2.LINE_8)

    for h in seg[1:]:
        if isinstance(h, list) and len(h) >= 6:
            hole = flatten_to_xylist(h)
            if hole.shape[0] >= 3:
                hole = clip_poly(hole, W, H)
                cv2.fillPoly(m, [hole.reshape(-1, 1, 2)], 0, lineType=cv2.LINE_8)

    return m

def rasterize_instance_boundary(seg, H, W, thickness=1):
    """Boundary pixels of ext + holes."""
    b = np.zeros((H, W), dtype=np.uint8)

    ext = flatten_to_xylist(seg[0])
    if ext.shape[0] >= 2:
        ext = clip_poly(ext, W, H)
        cv2.polylines(
            b, [ext.reshape(-1, 1, 2)], True, 1,
            thickness=thickness, lineType=cv2.LINE_8
        )

    for h in seg[1:]:
        if isinstance(h, list) and len(h) >= 4:
            hole = flatten_to_xylist(h)
            if hole.shape[0] >= 2:
                hole = clip_poly(hole, W, H)
                cv2.polylines(
                    b, [hole.reshape(-1, 1, 2)], True, 1,
                    thickness=thickness, lineType=cv2.LINE_8
                )

    b[b > 0] = 1
    return b

def dilate(bin_mask, delta):
    if delta <= 0:
        return bin_mask
    k = 2 * delta + 1
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(bin_mask.astype(np.uint8), kernel, iterations=1)

# Metric helpers.
def accumulate(stats, pred, gt):
    p = pred.astype(bool)
    g = gt.astype(bool)
    stats["TP"] += int(np.logical_and(p, g).sum())
    stats["FP"] += int(np.logical_and(p, ~g).sum())
    stats["FN"] += int(np.logical_and(~p, g).sum())
    stats["TN"] += int(np.logical_and(~p, ~g).sum())

def compute_metrics(stats):
    TP, FP, FN, TN = stats["TP"], stats["FP"], stats["FN"], stats["TN"]
    denom_all = TP + FP + FN + TN
    PA = (TP + TN) / denom_all if denom_all > 0 else 0.0
    F1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    IoU = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    return PA, F1, IoU

# Main entry.
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)

    # `coco_path` can be a file or a folder.
    parser.add_argument("--coco_path", type=str, default=None,
                        help="COCO GT JSON file or a folder containing multiple GT JSON files with images tables.")

    parser.add_argument("--gt_ext", type=str, default=None,
                        help="Optional GT extension override, e.g. .png. If omitted, use file_name as-is.")

    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)

    parser.add_argument("--delta", type=int, default=5)
    parser.add_argument("--line_thickness", type=int, default=1)

    parser.add_argument("--ignore_label", type=int, default=-1,
                        help="Ignore label in GT, e.g. 255. Use -1 to disable ignore handling.")
    parser.add_argument("--verbose", action="store_true", help="Print per-image debug info")

    args = parser.parse_args()
    H, W = args.height, args.width

    # Load predictions.
    per_image_per_class, id2name_pred = load_per_image_instances(args.pred_dir)
    image_ids = sorted(per_image_per_class.keys())
    if not image_ids:
        raise RuntimeError("No predictions found in pred_dir.")

    # Collect valid class keys.
    class_keys = sorted({k for v in per_image_per_class.values() for k in v})
    class_keys = [k for k in class_keys if infer_cat_id_from_stem(k) is not None]
    if not class_keys:
        raise RuntimeError("No valid class keys found (cannot map stems to NAME2ID).")

    # Build image_id -> file_name mapping from GT first, then predictions.
    id2name = {}
    if args.coco_path is not None:
        id2name.update(load_id2name_from_coco_path(args.coco_path))
    # Fill missing names from the prediction metadata when available.
    for iid, fn in id2name_pred.items():
        if iid not in id2name:
            id2name[iid] = fn

    # Prepare per-class metric accumulators.
    mask_stats = {k: dict(TP=0, FP=0, FN=0, TN=0) for k in class_keys}
    topo_stats = {k: dict(TP=0, FP=0, FN=0, TN=0) for k in class_keys}

    # Coverage accumulators help diagnose empty predictions or label mismatches.
    pred_area_sum = {k: 0 for k in class_keys}
    gt_area_sum = {k: 0 for k in class_keys}

    n_total = len(image_ids)
    n_no_name = 0
    n_no_gt = 0
    n_eval = 0

    for image_id in image_ids:
        if image_id not in id2name:
            n_no_name += 1
            continue

        file_name = os.path.basename(id2name[image_id])
        if args.gt_ext is not None:
            base = os.path.splitext(file_name)[0]
            gt_name = base + args.gt_ext
        else:
            gt_name = file_name

        gt_path = os.path.join(args.gt_dir, gt_name)
        if not os.path.exists(gt_path):
            n_no_gt += 1
            if args.verbose:
                print(f"[WARN] GT missing: {gt_path} (image_id={image_id})")
            continue

        gt_L = load_gt_mask(gt_path)
        if gt_L.shape[:2] != (H, W):
            raise ValueError(f"GT size mismatch for {gt_path}: got {gt_L.shape}, expected {(H,W)}")

        ignore = (gt_L == args.ignore_label) if args.ignore_label >= 0 else None
        valid = (~ignore) if (ignore is not None) else None

        n_eval += 1

        # Build per-image predicted unions on the fly for each class.
        for k in class_keys:
            cat_id = infer_cat_id_from_stem(k)
            if cat_id is None:
                continue

            gt_bin = (gt_L == cat_id).astype(np.uint8)

            pred_m = np.zeros((H, W), dtype=np.uint8)
            pred_b = np.zeros((H, W), dtype=np.uint8)

            for seg in per_image_per_class[image_id].get(k, []):
                pred_m |= rasterize_instance_mask(seg, H, W)
                pred_b |= rasterize_instance_boundary(seg, H, W, thickness=args.line_thickness)

            # Accumulate coverage statistics.
            if valid is not None:
                pred_area_sum[k] += int(pred_m[valid].sum())
                gt_area_sum[k] += int(gt_bin[valid].sum())
            else:
                pred_area_sum[k] += int(pred_m.sum())
                gt_area_sum[k] += int(gt_bin.sum())

            # Pixel-wise mask metrics.
            if valid is not None:
                accumulate(mask_stats[k], pred_m[valid], gt_bin[valid])
            else:
                accumulate(mask_stats[k], pred_m, gt_bin)

            # Topology metrics: convert boundaries and dilate by delta.
            gt_boundary = cv2.morphologyEx(
                gt_bin, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
            )
            gt_boundary[gt_boundary > 0] = 1

            pred_topo = dilate(pred_b, args.delta)
            gt_topo = dilate(gt_boundary, args.delta)

            if valid is not None:
                accumulate(topo_stats[k], pred_topo[valid], gt_topo[valid])
            else:
                accumulate(topo_stats[k], pred_topo, gt_topo)

    print(f"\n[INFO] images in predictions: {n_total}")
    print(f"[INFO] evaluated images    : {n_eval}")
    print(f"[INFO] missing id2name     : {n_no_name}")
    print(f"[INFO] missing GT file     : {n_no_gt}")

    if n_eval == 0:
        raise RuntimeError(
            "No images evaluated.\n"
            "- If predictions are annotation lists (no COCO 'images'), you MUST provide --coco_path.\n"
            "- Ensure GT files exist in --gt_dir and names match file_name.\n"
            "- Optionally set --gt_ext .png to force GT extension."
        )

    # Report per-class metrics and macro averages.
    print("\n=== Per-class metrics (dataset-accumulated) ===")
    mask_list, topo_list = [], []

    for k in class_keys:
        PA, F1, IoU = compute_metrics(mask_stats[k])
        PA2, F12, IoU2 = compute_metrics(topo_stats[k])

        mask_list.append({"PA": PA, "F1": F1, "IoU": IoU})
        topo_list.append({"PA": PA2, "F1": F12, "IoU": IoU2})

        # Zero GT coverage may indicate an absent class or a label-mapping issue.
        print(f"[{k}]  "
              f"PAmask={PA:.4f}  F1mask={F1:.4f}  mIoUmask={IoU:.4f} | "
              f"PAtopo={PA2:.4f}  F1topo={F12:.4f}  mIoUtopo={IoU2:.4f}   "
              f"(GT_pix={gt_area_sum[k]}, Pred_pix={pred_area_sum[k]})")

    print("\n=== Macro average over classes ===")
    print(f"PAmask  : {np.mean([d['PA'] for d in mask_list]):.4f}")
    print(f"F1mask  : {np.mean([d['F1'] for d in mask_list]):.4f}")
    print(f"mIoUmask: {np.mean([d['IoU'] for d in mask_list]):.4f}")
    print(f"PAtopo  : {np.mean([d['PA'] for d in topo_list]):.4f}")
    print(f"F1topo  : {np.mean([d['F1'] for d in topo_list]):.4f}")
    print(f"mIoUtopo: {np.mean([d['IoU'] for d in topo_list]):.4f}")

if __name__ == "__main__":
    main()


    """
    python eval_mask_topo_metrics.py \
  --pred_dir ./ACPV-Net/outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/poly_pslg_dp-2.5_angle_tol-10_corner_eps-2_replace_thresh-5_nms-3_th-0.5_topk-1k/categories \
  --coco_path ../work4/ACPV-Net/data/deventer_512/clean_unknown_artificial/categories_coco_ann \
  --width 512 --height 512 \
  --delta 5 \
  --line_thickness 1 \
  --gt_ext .png

    """
