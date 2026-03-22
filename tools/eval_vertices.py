"""
Evaluate vertex-level precision/recall (TP/FP/FN) from pre-extracted vertices.

Inputs:
  - pred_dir: directory containing predicted vertices in JSON files, e.g. [[x,y], ...]
  - gt_dir:   directory containing GT vertices in JSON files, e.g. [[x,y], ...] or [[x,y, ...], ...]

Matching:
  - one-to-one nearest neighbor with distance threshold (pixels)
"""

import os
import json
import numpy as np
from tqdm import tqdm
import argparse
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate vertex precision/recall from predicted vertex JSONs.")
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory with predicted vertices (.json)')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory with GT vertices (.json)')
    parser.add_argument('--dist_thresh', type=float, default=5.0, help='Matching distance threshold in pixels')
    parser.add_argument('--ext', type=str, default=".json", help='File extension to scan (default: .json)')
    parser.add_argument('--strict_intersection', action='store_true',
                        help='If set, only evaluate files that exist in BOTH pred_dir and gt_dir.')
    parser.add_argument('--save_per_image', type=str, default=None,
                        help='Optional: path to save per-image metrics as a JSON file.')
    return parser.parse_args()


def _load_vertices_generic(json_path):
    """
    Load vertices from a JSON file.

    Supported formats:
      - [[x, y], ...]
      - [[x, y, ...], ...]  (extra fields ignored)
      - {"vertices": [[x,y], ...]}  (if you ever used a dict wrapper)

    Returns:
      List[[x, y]] as integers.
    """
    if not os.path.exists(json_path):
        return []

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Optional dict wrapper support
    if isinstance(data, dict):
        if "vertices" in data:
            data = data["vertices"]
        else:
            raise ValueError(f"Unsupported JSON dict format in {json_path}: keys={list(data.keys())}")

    if not isinstance(data, list):
        raise ValueError(f"Unsupported JSON format in {json_path}: expected list, got {type(data)}")

    vertices = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        x, y = item[0], item[1]
        vertices.append([int(round(x)), int(round(y))])
    return vertices


def load_pred_vertices(json_path):
    return _load_vertices_generic(json_path)


def load_gt_vertices(json_path):
    return _load_vertices_generic(json_path)


def match_vertices(preds, gts, dist_thresh):
    """
    One-to-one matching between predicted and GT vertices using nearest neighbor search.

    Returns:
        tp, fp, fn
    """
    if len(preds) == 0 or len(gts) == 0:
        return 0, len(preds), len(gts)

    gts_np = np.asarray(gts, dtype=np.float32)
    tree = cKDTree(gts_np)

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

    pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith(args.ext)])
    gt_files = set([f for f in os.listdir(args.gt_dir) if f.endswith(args.ext)])

    if args.strict_intersection:
        eval_files = [f for f in pred_files if f in gt_files]
    else:
        # Evaluate all pred files; missing gt will be treated as empty GT
        eval_files = pred_files

    if len(eval_files) == 0:
        raise RuntimeError("No files found to evaluate. Check pred_dir/gt_dir and ext.")

    all_tp, all_fp, all_fn = 0, 0, 0
    per_image = {}

    for fname in tqdm(eval_files, desc="Evaluating"):
        pred_path = os.path.join(args.pred_dir, fname)
        gt_path = os.path.join(args.gt_dir, fname)

        preds = load_pred_vertices(pred_path)
        gts = load_gt_vertices(gt_path)  # if missing -> []

        tp, fp, fn = match_vertices(preds, gts, dist_thresh=args.dist_thresh)

        all_tp += tp
        all_fp += fp
        all_fn += fn

        # per-image metrics
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        per_image[fname] = {
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": float(prec), "recall": float(rec),
            "n_pred": int(len(preds)), "n_gt": int(len(gts)),
        }

    precision = all_tp / (all_tp + all_fp + 1e-8)
    recall = all_tp / (all_tp + all_fn + 1e-8)

    print("\n=== Vertex Evaluation Summary ===")
    print(f"Files evaluated: {len(eval_files)}")
    print(f"TP: {all_tp} | FP: {all_fp} | FN: {all_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    if args.save_per_image:
        os.makedirs(os.path.dirname(args.save_per_image), exist_ok=True) if os.path.dirname(args.save_per_image) else None
        with open(args.save_per_image, "w") as f:
            json.dump(per_image, f, indent=2)
        print(f"\nSaved per-image metrics to: {args.save_per_image}")


if __name__ == "__main__":
    main()

    """
    python eval_vertices.py \
  --pred_dir ./outputs/legacy/deventer_vmamba-small_512_vh_m_ldm_v1.5_simp_gt_b8/deventer_512/ddim/vertices_nms-3_th-0.5_topk-1k \
  --gt_dir ./data/deventer_512/poly_gt_global_boundary/test/d-2_angle_tol_deg-10-5-2_corner_eps-2_min_sep-3_len_px-10-6_v5/junctions \
  --dist_thresh 5
    """
